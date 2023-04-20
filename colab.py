if __name__ == "__main__":
    from pyngrok import ngrok
    from uvicorn import Config, Server

    config = Config("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
    server = Server(config)

    public_url = ngrok.connect(8000).public_url
    print(public_url)

    server.run()