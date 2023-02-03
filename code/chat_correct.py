import json

path = '/home/eva/code/email_project/data/'
f = open(path + "chat_new.json",'r')
chat = json.load(f)
x,y = 0,0
for e_i in chat:
    if "\n" in e_i or "\t" in e_i:
        x += 1
    for i, content in enumerate(chat[e_i]):
        if "\n" in content[0] or "\t" in content[0]:
            y += 1
            content[0] = content[0].replace("\n", "")
            content[0] = content[0].replace("\t", "")
            if "\n" in content[0] or "\t" in content[0]:
                print(1)
json_people = json.dumps(chat)
f = open(path + "chat_new.json","w")
f.write(json_people)
f.close()