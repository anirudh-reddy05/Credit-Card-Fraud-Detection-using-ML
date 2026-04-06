from app import create_app
from io import BytesIO
import os

app = create_app()

with app.test_client() as c:
    # open the existing instance user_data.csv and post it as a file
    csv_path = os.path.join(app.instance_path, 'user_data.csv')
    if not os.path.exists(csv_path):
        print('No user_data.csv found at', csv_path)
    else:
        data = {
            'file': (open(csv_path, 'rb'), 'user_data.csv')
        }
        response = c.post('/upload_page.html', data=data, content_type='multipart/form-data', follow_redirects=True)
        print('Status code:', response.status_code)
        print('Response data (truncated):')
        print(response.data.decode('utf-8')[:1000])
