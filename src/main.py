from dynaconf import settings

if __name__ == '__main__':
    print(settings.get("data_dir"))