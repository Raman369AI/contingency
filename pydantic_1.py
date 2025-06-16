#python -m venv myenv
#source myenv/bin/activate
#pip install pydantic[email]
#pip install opencv-python
'''Pydantic helps with the dynamic typing problem of python by strict type checking
- define clear and secure data models, 
eliminates error checks and boiler plate code making it reliable.'''

import pydantic
print(pydantic.__version__)

'''Type annotations are present in python but not enforced in python.'''


from pydantic import BaseModel
class User(BaseModel):
    id: int
    name:str = 'John Doe' #default value

user = User(id = 123)
print(user.model_fields_set)

#model_fields_set, doesnt display the default values
# print(user.model_dump())
# print(user.model_dump_json())
# print(user.model_json_schema())


#Nested Model
from typing import Optional, Annotated, Any
class Food(BaseModel):
    name: str
    price: float
    ingredients: Optional[list[str]] = None


class Restaurant(BaseModel):
    name: str
    location: str = 'New York'
    foods: list[Food]

vender = Restaurant(name = 'McDonalds', foods = [{'name': 'Burger', 'price': 3.50}, {'name': 'Fries', 'price': 2.00}])
# print(vender)
# print(vender.model_dump_json())

from pydantic import EmailStr, PositiveInt, Field, HttpUrl
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: PositiveInt

class Employee(BaseModel):
    name: str = Field(..., pattern = r"^[A-Za-z\s]+$")
    email: EmailStr
    position: str
    addresses: Annotated[list[Address], Field(min_length=2)]
    address : Address



# try:
#     employee = Employee(
#     name="John Doe",
#     email="john.doe@example.com",
#     position="Engineer",
#     addresses=[
#         {"street": "123 Main St", "city": "Springfield", "state": "IL", "zip": 62704},
#         {"street": "456 Elm St", "city": "Springfield", "state": "IL", "zip": 62705}
#     ],
#     address={"street": "123 Main St", "city": "Springfield", "state": "IL", "zip": 62704}
# )
#     print(employee)
# except pydantic.ValidationError as e:
#     print(e.errors())


from pydantic import field_validator, ValidationError, model_validator

class Owner(BaseModel):
    name: str
    email: EmailStr

    @field_validator('name')
    @classmethod
    def validate_name(cls, value: str) -> str:
        if len(value) < 2:
            raise ValueError('Name must be at least 2 characters long')
        return value.title()

# try:
#     owner = Owner(name="J", email="j@example.com")
# except ValidationError as e:
#     print(e.errors())

# try:
#     owner = Owner(name="John", email="j@example.com")
#     print(owner)
# except ValidationError as e:
#     print(e.errors())
from typing_extensions import Self
class Owner(BaseModel):
    name: str
    email: EmailStr

    @model_validator(mode='before')
    #Run before the instance of the class is instantiated.
    @classmethod
    def check_sensitive_omitted(cls,data: Any) -> Any:
        if isinstance(data, dict):
            if 'password' in data:
                raise ValueError('Password should not be included')
            if 'card_number' in data:
                raise ValueError('Card number should not be included')
        return data
    
    #Check the model after its creation  and before is for checking for the errors before they occur.
    @model_validator(mode = 'after')
    def check_name_contains_space(self) -> Self:
        if ' ' not in self.name:
            raise ValueError('Name should contain a space')
        return self

# try:
#     owner = Owner(name="John Doe", email="j@example.com")
#     print(owner)
# except ValidationError as e:
#     print(e.errors())

#to create our own names but use alias if the SQL database has some other name.
class User(BaseModel):
    name: str = Field(default = "Raman", alias = 'username')

user = User(username = 'RamanRah')

print(user)


from pydantic import computed_field
from datetime import datetime
class Person(BaseModel):
    name: str
    birth_year: int

    @computed_field
    @property
    def age(self) -> int:
        current_year = datetime.now().year
        a = current_year - self.birth_year
        return a
    
    

person = Person(name="John Doe", birth_year=1992)
# print(person.model_dump_json())
# print(person.age)

from pydantic import ValidationError
class User(BaseModel):
    name: str
    age: int

try:
    User.model_validate({'name': 'John Doe', 'age': 'raman'})
except ValidationError as e:
    print('Enter the right data type')

from pydantic import BaseModel, field_validator, ValidationError

ALLOWED_FRUITS = ['apple', 'banana', 'orange']

class FruitModel(BaseModel):
    fruit: str

    @field_validator('fruit')
    @classmethod
    def check_fruit(cls, v):
        if v not in ALLOWED_FRUITS:
            raise ValueError(f"Fruit must be one of {ALLOWED_FRUITS}")
        return v

FruitModel(fruit='apple')




