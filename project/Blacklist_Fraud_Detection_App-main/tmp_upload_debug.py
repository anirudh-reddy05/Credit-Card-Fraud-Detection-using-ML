from app import create_app
import os

app = create_app()

with app.test_client() as c:
    csv_path = os.path.join(app.instance_path, 'user_data.csv')
    if not os.path.exists(csv_path):
        print('No user_data.csv found at', csv_path)
    else:
        with open(csv_path, 'rb') as f:
            data = {
                'file': (f, 'user_data.csv')
            }
            response = c.post('/debug_model', data=data, content_type='multipart/form-data', follow_redirects=True)
            print('Status code:', response.status_code)
            body = response.data.decode('utf-8', errors='ignore')
            # look for common error messages
            checks = [
                'Missing required columns',
                'Invalid data types',
                'Error processing file',
                'Please upload a CSV file',
                'No file selected',
                'Traceback',
                'Exception'
            ]
            found = False
            for chk in checks:
                if chk in body:
                    print('Found message:', chk)
                    # print a surrounding snippet
                    idx = body.find(chk)
                    start = max(0, idx-200)
                    end = min(len(body), idx+400)
                    print(body[start:end])
                    found = True
            if not found:
                print('No known error strings found in response. Showing first 2000 chars of body:')
                print(body[:2000])
