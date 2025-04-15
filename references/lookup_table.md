## **Analysis of the Taxi Zone Lookup Table**

The `taxi_zone_lookup.csv` file contains 265 rows, each representing a taxi zone in NYC with the following columns:

    - `LocationID`: The unique ID for each taxi zone (matches PULocationID in the trip data).

    - `Borough`: The borough where the zone is located (e.g., Manhattan, Queens, Bronx).

    - `Zone`: The specific name of the zone (e.g., "Alphabet City", "World Trade Center").

    - `service_zone`: The type of service zone (e.g., "Yellow Zone", "Boro Zone", "EWR").

### **Key Observations:**
    - **Manhattan Zones**:
        - We need to filter for zones where Borough == "Manhattan". This will give us the LocationIDs for Manhattan zones, which we can use to filter the trip data.

        - From the sample data, zones like LocationID=4 (Alphabet City), LocationID=261 (World Trade Center), LocationID=262 (Yorkville East), and LocationID=263 (Yorkville West) are in Manhattan.

    - **Missing Values**:
        - Borough, Zone, and service_zone have some missing values:
            - Borough: 1 missing (264 non-null out of 265).

            - Zone: 1 missing.

            - service_zone: 2 missing (263 non-null).

        - From the data, LocationID=263 and LocationID=264 have Borough="Unknown" and Borough=NaN, respectively, with corresponding missing Zone and service_zone values. These zones are likely not relevant for our analysis, as they don’t represent actual taxi zones in NYC.

    - **Memory Usage**:
        - The lookup table is small (8.4 KB), so loading it into memory is not a concern, even alongside the larger trip data.

    - **Validation**:
        - LocationID ranges from 1 to 265, which matches the range of PULocationID and DOLocationID in the trip data (e.g., max of 265 in .describe()).

        - We’ll filter out LocationID=263 and LocationID=264 since they don’t represent valid zones for our prediction task.

### **Action**:
    - Extract all LocationIDs where Borough == "Manhattan".

    - Use this list to filter the trip data in Step 1.2 (Aggregate Raw Data into Time-Series).

