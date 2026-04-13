import pandas as pd
import json

for name, fname in [('region1', 'Region_1_coordinates.csv'), ('region2', 'Region_2_coordinates.csv')]:
    df = pd.read_csv(f'/scratch/users/k22026807/masters/project/alignment/{fname}', comment='#')
    coords = df[['X', 'Y']].values.tolist()
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    print(f'{name}: x={min(xs):.1f}-{max(xs):.1f}, y={min(ys):.1f}-{max(ys):.1f}')
    
    out = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': [coords]},
            'properties': {}
        }]
    }
    with open(f'/scratch/users/k22026807/masters/project/alignment/{name}_xenium.geojson', 'w') as f:
        json.dump(out, f)
    print(f'Saved {name}_xenium.geojson')
