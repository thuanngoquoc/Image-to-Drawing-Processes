class PriorityHandler:
    def __init__(self, priority_map):
        self.priority_map = priority_map

    def sort_parts(self, class_name, parts_dict):
        if class_name in self.priority_map:
            ordered_parts = []
            base_order = self.priority_map[class_name]
            for p in base_order:
                if p in parts_dict:
                    ordered_parts.append((p, parts_dict[p]))
            return ordered_parts
        else:
            # class không phân bộ phận
            return [("whole_object", parts_dict["whole"])]

    def sort_objects(self, object_list):
        # object_list = [{"class_name":..., "id":..., "parts":{...}}, ...]
        # sắp xếp theo id cho class không phân bộ phận
        return sorted(object_list, key=lambda x: x["id"])
