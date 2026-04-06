from pyngrok import ngrok
print('Opening ngrok tunnel...')

t = ngrok.connect(5000, "http")
print('Public URL:', t.public_url)
input('Press Enter to close tunnel...')
