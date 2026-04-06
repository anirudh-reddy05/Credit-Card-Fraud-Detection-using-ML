from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

def create_quick_admin():
    app = create_app()
    with app.app_context():
        # Create admin user with hardcoded credentials
        admin = User(
            email='admin@admin.com',
            password=generate_password_hash('admin123', method='sha256'),
            first_name='Admin',
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin created successfully!")
        print("Email: admin@admin.com")
        print("Password: admin123")

if __name__ == '__main__':
    create_quick_admin()