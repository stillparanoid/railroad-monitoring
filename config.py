import configparser

config = configparser.ConfigParser()
config.read("config.ini")

API_KEY = config["DEFAULT"]["google_api_key"]
SEARCH_ENGINE_ID = config["DEFAULT"]["custom_search_engine_id"]
DATA_FOLDER = config["DEFAULT"]["path_to_data_folder"]