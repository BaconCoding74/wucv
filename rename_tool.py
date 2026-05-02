import os


items = os.listdir('resources')
filtered_item = [item for item in items if not item.startswith('wuwa_inventory_system') and os.path.isfile(os.path.join('resources', item))]
current_renamed_item_count = len(items) - len(filtered_item)

for item in filtered_item:
    current_renamed_item_count += 1
    old_path = os.path.join('resources', item)
    new_path = os.path.join('resources', f'wuwa_inventory_system_{current_renamed_item_count}.png')
    os.rename(old_path, new_path)

    print(f'Renamed {old_path} to {new_path}')