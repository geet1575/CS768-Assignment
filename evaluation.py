import argparse

################################################
#               IMPORTANT                      #
################################################
# 1. Do not print anything other than the ranked list of papers.
# 2. Do not forget to remove all the debug prints while submitting.




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    # print(args)

    ################################################
    #               YOUR CODE START                #
    ################################################




    # prepare a ranked list of papers like this:
    result = ['paper1', 'paper2', 'paper3', 'paperK']  # Replace with your actual ranked list


    ################################################
    #               YOUR CODE END                  #
    ################################################


    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()
