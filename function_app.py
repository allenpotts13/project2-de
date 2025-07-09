import io
import os
import time
from io import StringIO

import azure.functions as func
import polars
import psycopg2
import requests
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
API_URL = f"{os.getenv('API_URL')}"


def fetch_api_data():
    headers = {"Content-Type": "application/json"}
    response = requests.get(API_URL, headers=headers)
    response.raise_for_status()
    return response.json()


def load_api_data(data):
    headers = data[0]
    rows = data[1:]
    try:
        df = polars.DataFrame(dict(zip(headers, zip(*rows))))

        # Rename API fields to match database schema
        column_mapping = {
            "NAME": "name",
            "B01001_001E": "population",
            "state": "state_fips",
            "county": "county_fips",
        }
        df = df.rename(column_mapping)

        # Select only the columns we need
        db_columns = ["name", "population", "state_fips", "county_fips"]
        df = df.select(db_columns)

        # Convert population to integer safely
        df = df.with_columns([df["population"].cast(polars.Int32, strict=False)])

        # Drop rows with any missing required values
        df = df.filter(
            df["population"].is_not_null()
            & df["name"].is_not_null()
            & df["state_fips"].is_not_null()
            & df["county_fips"].is_not_null()
        )

        # Cast to strings, strip spaces, fill nulls with ""
        df = df.with_columns(
            [
                df[col].cast(str).fill_null("").str.strip_chars().alias(col)
                for col in db_columns
            ]
        )

        # Filter out rows where all fields are empty (just in case)
        df = df.filter(
            (df["name"].str.len_chars() > 0)
            | (df["population"].str.len_chars() > 0)
            | (df["state_fips"].str.len_chars() > 0)
            | (df["county_fips"].str.len_chars() > 0)
        )

        # Step 1: Write cleaned CSV to StringIO
        output = StringIO()
        df.write_csv(output, separator="\t", include_header=False)
        output.seek(0)

        # Step 2: Strip blank lines
        cleaned_lines = [
            line
            for line in output.getvalue().splitlines()
            if line.strip() and len(line.split("\t")) == 4
        ]
        clean_output = StringIO("\n".join(cleaned_lines) + "\n")

        # Step 3: Load to Postgres
        target_table = "census_county_population"
        pg_conn = get_psycopg2_connection()

        with pg_conn.cursor() as cursor:
            try:
                cursor.copy_from(
                    clean_output, target_table, columns=db_columns, sep="\t", null=""
                )
                pg_conn.commit()
                print(f" Loaded {len(cleaned_lines)} clean rows into {target_table}")
            except Exception as e:
                pg_conn.rollback()
                print(f" COPY error: {e}")
                raise

        pg_conn.close()

    except Exception as e:
        print(f" Failed to load API data: {e}")
        if "pg_conn" in locals():
            pg_conn.close()
        raise


def get_blob_service_client():
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        raise ValueError("AzureWebJobsStorage environment variable is not set.")
    return BlobServiceClient.from_connection_string(conn_str)


def get_psycopg2_connection():
    try:
        connection_string = (
            f"host={os.environ.get('POSTGRES_HOST')} "
            f"port={os.environ.get('POSTGRES_PORT')} "
            f"dbname={os.environ.get('POSTGRES_DB')} "
            f"user={os.environ.get('POSTGRES_USER')} "
            f"password={os.environ.get('POSTGRES_PASSWORD')}"
        )
        return psycopg2.connect(connection_string)
    except Exception as e:
        error_message = f"Failed to connect to PostgreSQL with psycopg2: {str(e)}"
        raise ValueError(error_message)


def extract_csv_data_from_blob(
    filename, relevant_columns, column_mapping, schema_overrides=None
):
    CONTAINER_NAME = "nppes"
    try:
        print(f"Starting chunked blob data extraction for file: {filename}")
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=filename
        )

        file_buffer = io.BytesIO()

        blob_stream = blob_client.download_blob()

        for chunk in blob_stream.chunks():
            file_buffer.write(chunk)

        # Reset buffer position to beginning
        file_buffer.seek(0)

        if relevant_columns == []:
            lazy_df = polars.scan_csv(file_buffer, schema_overrides=schema_overrides)
        else:
            lazy_df = (
                polars.scan_csv(file_buffer, schema_overrides=schema_overrides)
                .select(relevant_columns)
                .rename(column_mapping)
            )

        return lazy_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_parquet_data_from_blob(filename):
    CONTAINER_NAME = "nppes"
    relevant_columns = [
        "NPI",
        "Entity Type Code",
        "Provider Organization Name (Legal Business Name)",
        "Provider Last Name (Legal Name)",
        "Provider First Name",
        "Provider Middle Name",
        "Provider Name Prefix Text",
        "Provider Name Suffix Text",
        "Provider Credential Text",
        "Provider Other Organization Name",
        "Provider First Line Business Practice Location Address",
        "Provider Second Line Business Practice Location Address",
        "Provider Business Practice Location Address City Name",
        "Provider Business Practice Location Address State Name",
        "Provider Business Practice Location Address Postal Code",
        "Healthcare Provider Taxonomy Code_1",
        "Healthcare Provider Primary Taxonomy Switch_1",
        "Healthcare Provider Taxonomy Code_2",
        "Healthcare Provider Primary Taxonomy Switch_2",
        "Healthcare Provider Taxonomy Code_3",
        "Healthcare Provider Primary Taxonomy Switch_3",
        "Healthcare Provider Taxonomy Code_4",
        "Healthcare Provider Primary Taxonomy Switch_4",
        "Healthcare Provider Taxonomy Code_5",
        "Healthcare Provider Primary Taxonomy Switch_5",
        "Healthcare Provider Taxonomy Code_6",
        "Healthcare Provider Primary Taxonomy Switch_6",
        "Healthcare Provider Taxonomy Code_7",
        "Healthcare Provider Primary Taxonomy Switch_7",
        "Healthcare Provider Taxonomy Code_8",
        "Healthcare Provider Primary Taxonomy Switch_8",
        "Healthcare Provider Taxonomy Code_9",
        "Healthcare Provider Primary Taxonomy Switch_9",
        "Healthcare Provider Taxonomy Code_10",
        "Healthcare Provider Primary Taxonomy Switch_10",
        "Healthcare Provider Taxonomy Code_11",
        "Healthcare Provider Primary Taxonomy Switch_11",
        "Healthcare Provider Taxonomy Code_12",
        "Healthcare Provider Primary Taxonomy Switch_12",
        "Healthcare Provider Taxonomy Code_13",
        "Healthcare Provider Primary Taxonomy Switch_13",
        "Healthcare Provider Taxonomy Code_14",
        "Healthcare Provider Primary Taxonomy Switch_14",
        "Healthcare Provider Taxonomy Code_15",
        "Healthcare Provider Primary Taxonomy Switch_15",
    ]

    column_mapping = {
        "NPI": "npi",
        "Entity Type Code": "entity_type_code",
        "Provider Organization Name (Legal Business Name)": "provider_organization_name",
        "Provider Last Name (Legal Name)": "provider_last_name",
        "Provider First Name": "provider_first_name",
        "Provider Middle Name": "provider_middle_name",
        "Provider Name Prefix Text": "provider_name_prefix",
        "Provider Name Suffix Text": "provider_name_suffix",
        "Provider Credential Text": "provider_credential",
        "Provider Other Organization Name": "provider_other_organization_name",
        "Provider First Line Business Practice Location Address": "provider_location_address_1",
        "Provider Second Line Business Practice Location Address": "provider_location_address_2",
        "Provider Business Practice Location Address City Name": "provider_city",
        "Provider Business Practice Location Address State Name": "provider_state",
        "Provider Business Practice Location Address Postal Code": "provider_postal_code",
        "Healthcare Provider Taxonomy Code_1": "healthcare_provider_taxonomy_code_1",
        "Healthcare Provider Primary Taxonomy Switch_1": "healthcare_provider_primary_taxonomy_switch_1",
        "Healthcare Provider Taxonomy Code_2": "healthcare_provider_taxonomy_code_2",
        "Healthcare Provider Primary Taxonomy Switch_2": "healthcare_provider_primary_taxonomy_switch_2",
        "Healthcare Provider Taxonomy Code_3": "healthcare_provider_taxonomy_code_3",
        "Healthcare Provider Primary Taxonomy Switch_3": "healthcare_provider_primary_taxonomy_switch_3",
        "Healthcare Provider Taxonomy Code_4": "healthcare_provider_taxonomy_code_4",
        "Healthcare Provider Primary Taxonomy Switch_4": "healthcare_provider_primary_taxonomy_switch_4",
        "Healthcare Provider Taxonomy Code_5": "healthcare_provider_taxonomy_code_5",
        "Healthcare Provider Primary Taxonomy Switch_5": "healthcare_provider_primary_taxonomy_switch_5",
        "Healthcare Provider Taxonomy Code_6": "healthcare_provider_taxonomy_code_6",
        "Healthcare Provider Primary Taxonomy Switch_6": "healthcare_provider_primary_taxonomy_switch_6",
        "Healthcare Provider Taxonomy Code_7": "healthcare_provider_taxonomy_code_7",
        "Healthcare Provider Primary Taxonomy Switch_7": "healthcare_provider_primary_taxonomy_switch_7",
        "Healthcare Provider Taxonomy Code_8": "healthcare_provider_taxonomy_code_8",
        "Healthcare Provider Primary Taxonomy Switch_8": "healthcare_provider_primary_taxonomy_switch_8",
        "Healthcare Provider Taxonomy Code_9": "healthcare_provider_taxonomy_code_9",
        "Healthcare Provider Primary Taxonomy Switch_9": "healthcare_provider_primary_taxonomy_switch_9",
        "Healthcare Provider Taxonomy Code_10": "healthcare_provider_taxonomy_code_10",
        "Healthcare Provider Primary Taxonomy Switch_10": "healthcare_provider_primary_taxonomy_switch_10",
        "Healthcare Provider Taxonomy Code_11": "healthcare_provider_taxonomy_code_11",
        "Healthcare Provider Primary Taxonomy Switch_11": "healthcare_provider_primary_taxonomy_switch_11",
        "Healthcare Provider Taxonomy Code_12": "healthcare_provider_taxonomy_code_12",
        "Healthcare Provider Primary Taxonomy Switch_12": "healthcare_provider_primary_taxonomy_switch_12",
        "Healthcare Provider Taxonomy Code_13": "healthcare_provider_taxonomy_code_13",
        "Healthcare Provider Primary Taxonomy Switch_13": "healthcare_provider_primary_taxonomy_switch_13",
        "Healthcare Provider Taxonomy Code_14": "healthcare_provider_taxonomy_code_14",
        "Healthcare Provider Primary Taxonomy Switch_14": "healthcare_provider_primary_taxonomy_switch_14",
        "Healthcare Provider Taxonomy Code_15": "healthcare_provider_taxonomy_code_15",
        "Healthcare Provider Primary Taxonomy Switch_15": "healthcare_provider_primary_taxonomy_switch_15",
    }

    try:
        print(f"Starting chunked blob data extraction for file: {filename}")
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=filename
        )

        file_buffer = io.BytesIO()

        print("Downloading blob data...")
        blob_stream = blob_client.download_blob()
        print(f"Blob download complete. Data size: {len(blob_stream):,} bytes")

        for chunk in blob_stream.chunks():
            file_buffer.write(chunk)

        # Reset buffer position to beginning
        file_buffer.seek(0)

        print("Scanning Parquet...")
        lazy_df = (
            polars.scan_parquet(file_buffer)
            .select(relevant_columns)
            .rename(column_mapping)
        )

        return lazy_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_chunked_blob_data_to_postgres(lazy_df, target_table, chunk_size=100_000):
    try:
        print(f"Starting to load data to {target_table} in chunks of {chunk_size:,}")

        pg_conn = get_psycopg2_connection()
        chunk_count = 0
        total_rows_processed = 0

        while True:
            chunk_count += 1
            batch_df = lazy_df.slice(total_rows_processed, chunk_size).collect()
            if batch_df.is_empty():
                print("No more data to process")
                break

            # FIXME: For Demo Purposes Only
            if chunk_count == 6 and target_table == "nppes_providers":
                print("Demo data limit reached")
                break

            current_chunk_size = len(batch_df)

            # Convert to CSV in memory for COPY
            output = StringIO()
            batch_df.write_csv(output, separator="\t", include_header=False)
            output.seek(0)

            # Use COPY for blazing fast bulk insert
            with pg_conn.cursor() as cursor:
                try:
                    if chunk_count == 1:
                        # Truncate table on first chunk if needed using stored procedure
                        cursor.execute("CALL truncate_table(%s)", (target_table,))

                    cursor.copy_from(
                        output,
                        target_table,
                        columns=list(batch_df.columns),
                        sep="\t",
                    )
                    pg_conn.commit()
                    total_rows_processed += current_chunk_size
                    print(
                        f"[SUCCESS] Successfully loaded chunk {chunk_count} ({current_chunk_size:,} rows)"
                    )

                except Exception as e:
                    pg_conn.rollback()
                    print(f"Error loading chunk {chunk_count}: {e}")
                    raise

            if current_chunk_size < chunk_size:
                break

        pg_conn.close()
        print(
            f"COMPLETE: Successfully loaded all {total_rows_processed:,} rows to {target_table}"
        )

    except Exception as e:
        print(f"Failed to write DataFrame to Postgres: {e}")
        if "pg_conn" in locals():
            pg_conn.close()
        raise


def convert_df_to_csv(df):
    output = StringIO()
    df.write_csv(output)
    output.seek(0)
    return output.getvalue()


def upload_csv_to_azure_blob(filename, data):
    CONTAINER_NAME = "nppes"
    try:
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=filename
        )

        # Handle encoding more robustly
        if isinstance(data, str):
            # Use errors='replace' to handle problematic characters
            blob_data = data.encode("utf-8", errors="replace")
        else:
            blob_data = data

        blob_client.upload_blob(blob_data, overwrite=True)
        print(f"Successfully uploaded {filename} to Azure Blob Storage")
        return None
    except Exception as e:
        print(f"Upload error for {filename}: {e}")
        return str(e)


@app.route(route="NPPES_Data_Cleaning")
def NPPES_Data_Cleaning(req: func.HttpRequest) -> func.HttpResponse:
    start_time = time.time()  # Tick
    try:
        api_data = fetch_api_data()
        load_api_data(api_data)

        body = req.get_json()

        # First & Large Data Target
        parquet_target_file = body.get("parquet_target_file")
        nppes_providers_table = "nppes_providers"
        lazy_df_1 = extract_parquet_data_from_blob(parquet_target_file)
        if lazy_df_1 is not None:
            load_chunked_blob_data_to_postgres(
                lazy_df_1, target_table=nppes_providers_table, chunk_size=100_000
            )

        # Additional Data Target 1
        csv_target_file_1 = body.get("csv_target_file_1")
        zip_county_table = "zip_county"
        zip_county_relevant_columns = [
            "ZIP",
            "COUNTY",
            "USPS_ZIP_PREF_CITY",
            "USPS_ZIP_PREF_STATE",
            "RES_RATIO",
            "BUS_RATIO",
            "OTH_RATIO",
            "TOT_RATIO",
        ]
        zip_county_column_mapping = {
            "ZIP": "zip",
            "COUNTY": "county",
            "USPS_ZIP_PREF_CITY": "usps_zip_pref_city",
            "USPS_ZIP_PREF_STATE": "usps_zip_pref_state",
            "RES_RATIO": "res_ratio",
            "BUS_RATIO": "bus_ratio",
            "OTH_RATIO": "oth_ratio",
            "TOT_RATIO": "tot_ratio",
        }
        # Add schema overrides to ensure ZIP codes are treated as strings
        zip_county_schema_overrides = {
            "ZIP": polars.Utf8,
            "COUNTY": polars.Utf8,
            "USPS_ZIP_PREF_CITY": polars.Utf8,
            "USPS_ZIP_PREF_STATE": polars.Utf8,
        }
        lazy_df_2 = extract_csv_data_from_blob(
            csv_target_file_1,
            zip_county_relevant_columns,
            zip_county_column_mapping,
            zip_county_schema_overrides,
        )
        if lazy_df_2 is not None:
            load_chunked_blob_data_to_postgres(
                lazy_df_2, target_table=zip_county_table, chunk_size=100_000
            )

        # Additional Data Target 2
        csv_target_file_2 = body.get("csv_target_file_2")
        ssa_fips_state_county_table = "ssa_fips_state_county"
        ssa_relevant_columns = []
        ssa_column_mapping = {}
        ssa_schema_overrides = {
            "ssa_code": polars.Utf8,
            "fipscounty": polars.Utf8,
            "cbsa_code": polars.Utf8,
        }

        lazy_df_3 = extract_csv_data_from_blob(
            csv_target_file_2,
            ssa_relevant_columns,
            ssa_column_mapping,
            ssa_schema_overrides,
        )
        if lazy_df_3 is not None:
            load_chunked_blob_data_to_postgres(
                lazy_df_3, target_table=ssa_fips_state_county_table, chunk_size=100_000
            )

        # Additional Data Target 3
        csv_target_file_3 = body.get("csv_target_file_3")
        nucc_taxonomy_table = "nucc_taxonomy"
        taxonomy_relevant_columns = [
            "Code",
            "Grouping",
            "Classification",
            "Specialization",
        ]
        taxonomy_column_mapping = {
            "Code": "code",
            "Grouping": "grouping",
            "Classification": "classification",
            "Specialization": "specialization",
        }
        taxonomy_schema_overrides = {
            "Code": polars.Utf8,
            "Grouping": polars.Utf8,
            "Classification": polars.Utf8,
            "Specialization": polars.Utf8,
        }

        lazy_df_4 = extract_csv_data_from_blob(
            csv_target_file_3,
            taxonomy_relevant_columns,
            taxonomy_column_mapping,
            taxonomy_schema_overrides,
        )
        if lazy_df_4 is not None:
            load_chunked_blob_data_to_postgres(
                lazy_df_4, target_table=nucc_taxonomy_table, chunk_size=100_000
            )

        # Run data cleaning and transformation after all raw data is loaded
        print("Running data cleaning and transformation...")
        pg_conn = get_psycopg2_connection()
        with pg_conn.cursor() as cursor:
            # Execute stored procedures using psycopg2
            cursor.execute("CALL clean_and_populate_nppes_data()")
            cursor.execute("CALL create_export_view()")
            pg_conn.commit()
        pg_conn.close()
        print("Data cleaning and transformation completed")

        # Optional: Export clean data to CSV (can be controlled via request parameter)
        export_csv = body.get("export_csv", True)  # Default to True
        if export_csv:
            print("Starting CSV export of clean data...")
            csv_filename = body.get("csv_filename", "nppes_clean_export.csv")
            csv_chunk_size = body.get("csv_chunk_size", 50000)

            export_success = export_clean_data_to_csv_chunked(
                csv_chunk_size, csv_filename
            )
            if export_success:
                print(f"CSV export completed: {csv_filename}")
            else:
                print("CSV export failed, but continuing...")

        elapsed = time.time() - start_time  # Tock
        response = f"Data processing completed in {elapsed:.2f} seconds"
        return func.HttpResponse(response, status_code=200)
    except Exception as e:
        error_message = f"Internal server error: {str(e)}"
        return func.HttpResponse(error_message, status_code=500)


def export_clean_data_to_csv_chunked(
    chunk_size=50000, output_filename="nppes_clean_export.csv"
):
    """
    Export clean NPPES data to CSV - simple and efficient.
    """
    try:
        print(f"Starting CSV export to file: {output_filename}")

        pg_conn = get_psycopg2_connection()

        # Initialize CSV content
        csv_content = []
        processed_count = 0

        # Add headers
        headers = [
            "npi",
            "entity_type",
            "entity_name",
            "provider_location_address_1",
            "provider_location_address_2",
            "provider_city",
            "provider_state",
            "provider_postal_code_clean",
            "county_name",
            "state_name",
            "primary_taxonomy_code",
            "taxonomy_grouping",
            "taxonomy_classification",
            "taxonomy_specialization",
            "data_quality_score",
        ]
        csv_content.append(",".join(f'"{col}"' for col in headers))

        # Export data in chunks
        chunk_start = 0
        while True:
            print(f"Processing chunk starting at record {chunk_start:,}")

            with pg_conn.cursor() as cursor:
                # Get chunked data from the stored function
                cursor.execute(
                    "SELECT npi, entity_type, entity_name, provider_location_address_1, provider_location_address_2, provider_city, provider_state, provider_postal_code_clean, county_name, state_name, primary_taxonomy_code, taxonomy_grouping, taxonomy_classification, taxonomy_specialization, data_quality_score FROM get_export_chunk(%s, %s)",
                    (chunk_size, chunk_start),
                )
                rows = cursor.fetchall()

                if not rows:
                    print("No more records found. Export complete.")
                    break

                print(f"Retrieved {len(rows)} rows in this chunk")

                # Convert rows to CSV format
                for row in rows:
                    escaped_row = []
                    for i, value in enumerate(row):
                        if value is None:
                            escaped_row.append('""')
                        else:
                            # Special handling for ZIP codes to preserve leading zeros
                            if (
                                headers[i] == "provider_postal_code_clean"
                                and value is not None
                            ):
                                str_value = str(value).zfill(5)
                            else:
                                str_value = str(value)

                            # Escape quotes and wrap in quotes
                            str_value = str_value.replace('"', '""')
                            escaped_row.append(f'"{str_value}"')
                    csv_content.append(",".join(escaped_row))

                processed_count += len(rows)
                chunk_start += chunk_size
                print(f"Processed {processed_count:,} records so far")

                # If we got less than the chunk size, we're done
                if len(rows) < chunk_size:
                    break

        pg_conn.close()

        # Join all CSV content
        final_csv = "\n".join(csv_content)

        # Upload to Azure Blob
        print(f"Uploading CSV file: {output_filename}")
        upload_success = upload_csv_to_azure_blob(output_filename, final_csv)

        if upload_success is None:
            print(
                f"Successfully exported {processed_count:,} records to {output_filename}"
            )
            return True
        else:
            print("Upload failed")
            return False

    except Exception as e:
        print(f"Error during CSV export: {e}")
        if "pg_conn" in locals():
            pg_conn.close()
        return False


@app.route(route="export_clean_csv")
def export_clean_csv(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function endpoint to export clean NPPES data to CSV
    """
    start_time = time.time()

    try:
        # Get optional parameters from request
        body = req.get_json() if req.get_body() else {}
        chunk_size = body.get("chunk_size", 50000)
        output_filename = body.get("output_filename", "nppes_clean_export.csv")

        print(
            f"Starting CSV export with chunk_size={chunk_size}, filename={output_filename}"
        )

        # Run the export
        success = export_clean_data_to_csv_chunked(chunk_size, output_filename)

        elapsed = time.time() - start_time

        if success:
            response_message = f"CSV export completed successfully in {elapsed:.2f} seconds. File: {output_filename}"
            return func.HttpResponse(response_message, status_code=200)
        else:
            return func.HttpResponse(
                "CSV export failed. Check logs for details.", status_code=500
            )

    except Exception as e:
        elapsed = time.time() - start_time
        error_message = f"CSV export error after {elapsed:.2f} seconds: {str(e)}"
        return func.HttpResponse(error_message, status_code=500)
