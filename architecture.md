# The architecture of `piet-gpu`

`piet` is a 2D graphics API. `piet-gpu` is an experimental GPU accelerated backend for `piet`, based on using general purpose GPU computing to handle 2D rendering tasks. 

## The pipeline

To use a GPU effectively for a computational task, one should first break down the task into a series of sequential sub-tasks (the "pipeline"). Each sub-task should be chosen so that it can easily be broken into primitive tasks that are easy to parallelize. 

For example, in the final sub-task of 3D rendering, the parallelizable primitve is the computation rendering a triangle. A triangle can be thought of as the primitive data structure of this stage.

In `piet-gpu`'s final rendering stage, the parallelizable primitive is not a triangle. [Possible ref to Lyon, which tries triangularization?]. Instead, the surface to which rendering is output is tiled by square tiles. Each tile is the primitive data structure, and tile rendering, the primitive task, can be performed independently of any other tile, making it trivial to parallelize. [Possible ref to Pathfinder docs/info regarding why tiling is a good primitive for 2D rendering.] 

The overall pipeline of sub-tasks of `piet-gpu` is:

![](./pipeline.png)

* (CPU) An application defines a "scene" using `piet`'s API. The scene has a graph-like structure, and essentially represents the operations (translation, rotation, etc.) that should be applied to drawing commands (child vertices). 
* (CPU) The scene is encoded into a compact format, then transmitted to the GPU.
* (GPU) The scene is re-interpreted into per-tile-command-lists (PTCLs) of compound or simple drawing commands. Compound drawing commands are processed further to break them down into  simple drawing commands.
* (GPU) Tiles are rendered in parallel, by executing the simple drawing commands stored in their command lists. 

The CPU/GPU label given to each pipeline stage describes whether it is executed on the CPU or the GPU. Any task marked GPU must have been broken down into primtive tasks in some manner, so that it can be parallelized. The rest of this document is the story of how that's done.

## Encoding/decoding of scenes

A scene is a list of variants of the `PietItem` enum. These variants are:
* structures representing drawing commands (circle, stroke, etc.)
* structures containing metadata that a) groups subsequent `PietItem` in the list (i.e. "the next N items are in this group), and b) specifies the transform to be applied to all items in the group 

GPU-side, shader languages do not provide enums. So we decided to write Rust macros which generate GLSL functions which can interpret the scene buffer as if it was a list of enums.

We also want to encode a scene compactly in the number of bytes used. This is for two reasons: 1) to optimize upload of data from the CPU to the GPU, and 2) to leverage memory locality (doing a bunch of lookups for data which are tightly grouped in a particular region of a buffer is much faster than doing a bunch of lookups for data strewn all across the buffer). Encoding the scene compactly means that we will also have to generate GLSL functions which can decode data in the scene.

Summarizing, we wrote macros which:
* generate Rust functions that can encode (simple) Rust structures and enums into a compact-byte representation for transmission to the GPU;
* generate GLSL functions that can access/decode the compact-byte scene data into GPU-side structures; these functions emulate structure-field accesses
* generate GLSL functions that can pack GPU-side structure into a compact-byte representation (useful for passing data between stages on the GPU). 

Once a scene is encoded into a compact representation CPU-side, it is then transmitted to the GPU.

> This CPU-side scene encoding could be parallelized, but currently encoding is done purely sequentially, for ease of programming.

## Generating Simple Drawing Commands from a scene (Kernel 1 and Kernel 2)

To generate drawing commands from the scene, we must traverse its essentially graph-like structure. Kernel 1 does this and produces a list of drawing commands for groups of tiles. Some of these drawing commands might be "compound", i.e. requiring post-processing into ready-to-render "simple drawing commands". This post-processing is done in Kernel 2. 

### Kernel 1: Traversing the scene

Kernel 1 programs threads to traverse the scene, and make a list of drawing commands (with all relevant operations applied) that are relevant to groups of tiles ("tilegroup"). Note that one cannot simply walk through the scene as if it was a list, since it is interspersed with metadata grouping together items. In this sense, it has a nested/recursive/graph-like structure, so each thread will need to maintain a stack keeping track of items whose processing it has temporarily put on hold, in order to first process "lower-level" items. The output of kernel 1 is a list of relevant (intersecting with the tilegroup) drawing commands ("items") per tilegroup. 

Currently, Kernel 1 leaves many possible performance optimizations on the table. However, this means it is somewhat easy to understand at the moment. Here's the pseudocode:

```GLSL
// The input scene graph. 
layout("...") readonly buffer SceneBuf {
    uint[] scene;
};

// Output buffer.
layout("...") buffer TilegroupBuf {
    uint[] tilegroup;
};

// Extra memory to store output. Used as necessary.
layout("...") buffer AllocBuf {
    uint alloc;
}; 

// Each thread maintains stack of upstream vertices to revisit.
#define MAX_STACK 8

struct StackElement {
    PietItemRef group;
    // index of item within group        
    uint index;
    // group.offset.xy, why do we take it out?
    vec2 offset;
};

void main() {
    StackElement stack[MAX_STACK];
    uint stack_ix = 0;

    // abbreviate tilegroup as tg      
    uint tg_ix = determine_tilegroup_to_process(thread_id);
    // get ref to tilegroup's memory in tilegroup buf
    TileGroupRef tg_ref = get_ref_to_tg(tg_ix);
    // can bump up space allocated later, if needed
    uint tg_limit = determine_max_memory_allocated_for_tg();
    // abbreviate bounding box as bb
    // only need to get position of tilegroup bbox, as tilegroup bbox size is fixed
    vec2 xy0 = get_tg_bbox_corner(tg_ix);
 
    // root item in scene
    PietItemRef root = PietItemRef(0);
    // fetch root item from scene
    SimpleGroup group = PietItem_Group_read(root);
    StackElement tos = StackElement(root, 0, group.offset.xy);

    while (true) {
        if (tos.index < group.n_items) {
            Bbox bbox = get_bbox_of_item(tos.index);
            // object bbox is relative to position within group, but we want absolute bbox
            vec4 bb = vec4(bbox.bbox) + tos.offset.xyxy;
            // does bbox intersect with tilegroup bounds?
            bool hit = does_bbox_intersect_tg_bounds();
            bool is_group = false;
            // don't bother fetching item data, unless its bbox hits 
            if (hit) {       
                PietItemRef item_ref = PietItem_index(group.items, tos.index);
                is_group = PietItem_tag(item_ref) == PietItem_Group;
            }
            if (hit && !is_group) {
                PietItemRef item_ref = PietItem_index(group.items, tos.index);
                // instance is just an item's ref and its offset
                Instance ins = Instance(item_ref, tos.offset);
                
                if (tg_ref.offset > tg_limit) {
                    // Allocation exceeded; do atomic bump alloc.
                    set_up_link_to_start_of_memory_extension();
                }
                TileGroup_Instance_write(tg_ref, ins);
                tg_ref.offset += Instance_size;
            }
            if (is_group) {
                put_current_group_on_stack();
                set_item_as_current_group();
            } else {
                tos.index++;
            }
        } else {
            // processed all items in this group; pop the stack
            if (stack_ix == 0) {
                break;
            }
            tos = stack[--stack_ix];
            group = PietItem_Group_read(tos.group);
        }
    }
    TileGroup_End_write(tg_ref);
}
```

### Further processing of tilegroup items (Kernel 2)

Kernel 2 processes items in the tilegroup further. In particular, fills and strokes are broken down into simpler drawing commands.

```GLSL
// Pseudocode under construction.
```

## Generating PTCLs from output of Kernels 1 + 2 (Kernel 3)

Recall that the output of the first two kernels is a list of simple drawing commands per tilegroup. Ultimately, we are interested in producing a list of simple drawing commands per tile, and this is Kernel 3's job.

The code for Kernel 3, as you might imagine, is similar in spirit to the code for Kernel 1, but it is much simpler. In particular, Kernel 3 has a no-frills, sequential, list of drawing commands to walk through. Otherwise:
* in Kernel 1 we checked to see if the area of effect of each drawing command's from the scene buffer intersected with a tilegroup;
* instead, in Kernel 3, we check to see if the area of effect for each drawing command from the tilegroup buffer intersects with a tile's bounding box.

To recap, the output of Kernel 3 is a list of drawing commands per tile.

## Drawing (Kernel 4)

Kernel 4 finally gets to the work of painting pixels based on drawing commands. This time, we parallelize work at the level of the pixels in a tile. While traversing the PTCL for the relevant tile, each drawing command's effect on the pixel is calculated, and the result written out into a buffer representing the output surface. 