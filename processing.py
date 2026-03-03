import requests
import os

def retrieve_ais(bbox, start_year, stop_year, step=1, width=1600, height=693, crs=3995):
    '''
    
    
    Example link:
    https://gmtds.maplarge.com/ogc/ais:density/wms?REQUEST=GetMap&LAYERS=ais:density&STYLES=&FORMAT=image/png&TRANSPARENT=true
    &SERVICE=WMS&VERSION=1.3.0&WIDTH=1600&HEIGHT=693&CRS=EPSG:3995
    &BBOX=-10115103.215710418,-4227137.379356397,9208953.375633113,3393787.4388547074&TIME=2023-04-01T00:00:00Z
    &CQL_FILTER="category_column='All' AND category='All'"
    '''
    
    base = 'https://gmtds.maplarge.com/ogc/ais:density/wms'

    for year in range(start_year, stop_year+1, step=step):
        timestamp = f'{year}-09-01T00:00:00Z'
        
        params = {
            'REQUEST': 'GetMap',
            'LAYERS': 'ais:density',
            'STYLES': '',
            'FORMAT': 'image/geotiff',
            'TRANSPARENT': 'true',
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'WIDTH': width,
            'HEIGHT': height,
            'CRS': f'EPSG:{crs}',
            'BBOX': bbox,
            'TIME': timestamp,
            'CQL_FILTER': '"' + "category_column='All' AND category='All'" + '"'
        }

        filename = f'arctic_maritime_{year}'

        print(f'Requesting data for {year}:')
        try:
            response = requests.get(base, params=params, timeout=40)
            if response.status_code==200:
                with open(f'/data/ais_traffic/{filename}', 'wb') as file:
                    file.write(response.content)
            else:
                print(f'Error: {response.status_code}')
        except Exception as e:
            print(f'Failed: {e}')


