from flask.cli import with_appcontext
from werkzeug.security import generate_password_hash
from .models import User
from . import db
import click

def create_superuser_command():
    email = input('Enter email address: ')
    password = input('Enter password: ')
    first_name = input('Enter first name: ')

    try:
        if User.query.filter_by(email=email).first():
            print('Error: Email already exists')
            return

        user = User(
            email=email,
            first_name=first_name,
            password=generate_password_hash(password, method='sha256'),
            is_admin=True
        )
        
        db.session.add(user)
        db.session.commit()
        print(f'Superuser {email} created successfully!')
        
    except Exception as e:
        print(f'Error creating superuser: {str(e)}')
        db.session.rollback()