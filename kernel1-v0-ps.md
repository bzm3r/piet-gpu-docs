```GLSL
// The input scene graph. 
layout("...") readonly buffer SceneBuf {
    uint[] scene;
};

// Output buffer.
layout("...") buffer TilegroupBuf {
    uint[] tilegroup;
};

// Unknown
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