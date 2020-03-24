import sys
import paramiko

def main():
    username = sys.argv[1]
    password = sys.argv[2]

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('odioelodio.com', username=username)

    query = "use pgodio;"

    print("Command:", end=" ")
    queryCommand = input()
    print("")
    while queryCommand != "exit":
        if queryCommand == "hatefulCount":
            query += "select count(distinct tweet_id) from votesIsHateful;"
        elif queryCommand == "ambiguous":
            pass
        elif queryCommand == "skipped":
            pass

        stdin, stdout, stderr = client.exec_command(f'mysql -u test -p{password} -e "{query}"')
        for line in stdout:
            print(line)
        
        print("Command:", end=" ")
        queryCommand = input()
        print("")

    client.close()

main()