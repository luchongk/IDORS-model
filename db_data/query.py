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
        if queryCommand == "hateful":
            query += """select tweet_id, count(*) as cnt, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate, text
                        from votesIsHateful join tweets on id = tweet_id group by tweet_id having cnt_hate - cnt_not_hate > 1;"""
        elif queryCommand == "hatefulCount":
            query += """select count(*) from (select tweet_id, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate, text
                        from votesIsHateful join tweets on id = tweet_id group by tweet_id having cnt_hate - cnt_not_hate > 1) as t1;"""
        elif queryCommand == "nonhateful":
            query += """select tweet_id, count(*) as cnt, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate, text
                        from votesIsHateful join tweets on id = tweet_id group by tweet_id having cnt_not_hate - cnt_hate > 1;"""
        elif queryCommand == "nonhatefulCount":
            query += """select count(*) from (select tweet_id, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate, text
                        from votesIsHateful join tweets on id = tweet_id group by tweet_id having cnt_not_hate - cnt_hate > 1) as t1;"""
        elif queryCommand == "ambiguous":
            query += """select tweet_id, count(*) as cnt, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate, text
                        from votesIsHateful join tweets on id = tweet_id group by tweet_id having abs(cnt_hate - cnt_not_hate) <= 1;"""
        elif queryCommand == "ambiguousCount": 
            query += """select count(*) from (select tweet_id, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate
                        from votesIsHateful group by tweet_id having abs(cnt_hate - cnt_not_hate) <= 1) as t1;"""
        elif queryCommand == "offensive":
            query += """select tweet_id, count(*) as cnt, sum(is_hateful) as cnt_hate, count(*) - sum(is_hateful) as cnt_not_hate, sum(is_offensive), text
                        from votesIsHateful join tweets on id = tweet_id group by tweet_id having sum(is_offensive) > 0;"""
        elif queryCommand == "hateTypes":    
            query += """select tweet_id, hate_type, count(*), text
                        from tweets join votesHateType on tweets.id = tweet_id group by tweet_id, text, hate_type order by tweet_id;"""
        elif queryCommand == "skipped":
            query += "select id, skip_count, text from tweets where skip_count > 0 order by skip_count DESC;"
        elif queryCommand == "skippedCount":
            query += "select count(*) from (select id, skip_count from tweets where skip_count > 0) as t1;"
        elif queryCommand == "tweetCount":
            query += "select count(*) from tweets;"
        elif queryCommand == "totalVoteCount":
            query += "select count(*) from votesIsHateful;"
        elif queryCommand == "votedTweets":
            query += "select count(distinct tweet_id) from votesIsHateful;"

        if queryCommand == "help":
            help()
        else:
            stdin, stdout, stderr = client.exec_command(f'mysql -u test -p{password} -e "{query}"')
            for line in stdout:
                print(line, end="")
            print()
        
        query = "use pgodio;"
        print("Command:", end=" ")
        queryCommand = input()
        print("")

    client.close()

def help():
    print("List of commands:")
    print()
    print("- hateful")
    print("- hatefulCount")
    print("- nonhateful")
    print("- nonhatefulCount")
    print("- ambiguous")
    print("- ambiguousCount")
    print("- skipped")
    print("- skippedCount")
    print("- tweetCount")
    print("- totalVoteCount")
    print("- votedTweets")
    print()

main()