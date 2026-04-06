from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

def create_admin():
    app = create_app()
    with app.app_context():
        print("\n=== Create Admin User ===")
        email = input("Enter admin email: ")
        password = input("Enter admin password: ")
        first_name = input("Enter admin first name: ")

        try:
            # Check if user already exists
            if User.query.filter_by(email=email).first():
                print("Error: Email already exists!")
                return

            # Create new admin user
            admin = User(
                email=email,
                first_name=first_name,
                password=generate_password_hash(password, method='sha256'),
                is_admin=True
            )

            db.session.add(admin)
            db.session.commit()
            print(f"\nAdmin user '{email}' created successfully!")

        except Exception as e:
            print(f"Error creating admin user: {str(e)}")
            db.session.rollback()

if __name__ == '__main__':
    create_admin()