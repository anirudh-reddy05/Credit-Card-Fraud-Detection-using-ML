from app import create_app
import io

app = create_app()

csv_header = ','.join(['amt','zip','lat','long','genders','Essentials','Leisure','Wellness','Other'] + [
    'AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA',
    'MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI',
    'SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY'
])

# two different rows
row1 = ','.join(['6.38','74131','36.0557','-96.0602','1','0','1','0','0'] + ['0']*51)
row2 = ','.join(['1200','10001','40.7128','-74.0060','1','1','0','0','0'] + ['0']*51)

csv1 = csv_header + '\n' + row1 + '\n'
csv2 = csv_header + '\n' + row2 + '\n'

with app.test_client() as c:
    for i, csv_bytes in enumerate([csv1, csv2], start=1):
        data = {'file': (io.BytesIO(csv_bytes.encode('utf-8')), f'test{i}.csv')}
        resp = c.post('/debug_model', data=data, content_type='multipart/form-data', follow_redirects=True)
        body = resp.data.decode('utf-8', errors='ignore')
        # very small parser to extract adjusted_prob
        adj = None
        if 'Adjusted probability' in body:
            # find the section
            idx = body.find('Adjusted probability')
            start = body.find('<pre>', idx)
            end = body.find('</pre>', start)
            if start != -1 and end != -1:
                adj = body[start+5:end].strip()
        print('=== Test row', i, '===')
        print('Status:', resp.status_code)
        if adj:
            print('Adjusted prob:', adj)
        else:
            # fallback show snippet
            print(body[:800])

print('Done')
