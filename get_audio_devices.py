import sounddevice as sd


if __name__ == '__main__':
    for device in sd.query_devices():
        print(f"{device['index']}: {device['name']}")