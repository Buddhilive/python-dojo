print('Hello World')

a = 2
b = 5

def add():
    c = a + b
    print(c)

add()

print(type(a))

msg = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(msg)
print(msg[10])

def loopBanana():
  for x in "banana":
    print(x)

print(len(msg))

print("labore" in msg)

print(msg.upper())

print(msg.split(' '))

age = 36
txt = "My name is John, and I am {}"
print(txt.format(age))

txt = "Hello Sam!"
mytable = txt.maketrans("S", "P")
print(txt.translate(mytable))

thislist = ["apple", "banana", "cherry", "orange", "kiwi", "mango"]
thislist[1:3] = ["blackcurrant"]
print(thislist)
thislist.insert(2, "watermelon")
print(thislist)
thislist.pop(2)
print(thislist)
for x in thislist:
  print(x)
thislist.sort()
print(thislist)
def myfunc(n):
  return abs(n - 50)

thislist2 = [100, 50, 65, 82, 23]
thislist2.sort(key = myfunc)
print(thislist2)