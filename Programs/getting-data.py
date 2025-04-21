from astroquery.vizier import Vizier
import numpy as np
from astropy.table import vstack
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configurar sesión con reintentos
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)
Vizier.requests_session = session
Vizier.TIMEOUT = 60

Vizier.ROW_LIMIT = -1
v = Vizier(column_filters={"emitter": ">=1"})

lon_min = 30
lon_max = 215
step = 5  # Segmentos de 5° con solapamiento
glon_ranges = np.arange(lon_min, lon_max, step)  # No superar 215°

tables = []
for i, glon_start in enumerate(glon_ranges):
    glon_end = min(glon_start + step + 0.1, lon_max)  # Solapamiento de 0.1°
    print(f"Segmento {i+1}/{len(glon_ranges)}: l = {glon_start} a {glon_end}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            catalog = v.query_constraints(
                catalog="V/165",
                GLON=f"{glon_start} .. {glon_end}",
                GLAT="-5 .. 5"
            )
            if catalog:
                tables.append(catalog[0])
            time.sleep(2)
            break  # Éxito, salir del bucle de reintentos
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error en segmento {glon_start}-{glon_end}: {str(e)}")
            time.sleep(10)

merged_table = vstack(tables)
print(len(merged_table))
