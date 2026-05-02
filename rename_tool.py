import os

folder = 'test_cases'
starting_name = 'wuwa_inventory_system'

items = os.listdir(f'{folder}')
filtered_item = [item for item in items if not item.startswith(starting_name) and os.path.isfile(os.path.join(folder, item))]
current_renamed_item_count = len(items) - len(filtered_item)

for item in filtered_item:
    current_renamed_item_count += 1
    old_path = os.path.join(folder, item)
    new_path = os.path.join(folder, f'{starting_name}_{current_renamed_item_count}.png')
    os.rename(old_path, new_path)

    print(f'Renamed {old_path} to {new_path}')