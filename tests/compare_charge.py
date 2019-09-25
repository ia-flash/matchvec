# Example 3: asynchronous requests with larger thread pool
import asyncio
import concurrent.futures
import requests
import time

num_test = 5

def calling(_data):
    start_time = time.time()
    requests.post('https://t7p0v4a94l.execute-api.eu-west-1.amazonaws.com/Prod/post',
            _data)
    print("{:.3f}".format(time.time() - start_time))

def single_call():
    filepath = 'clio4.json'
    with open(filepath) as fh:
        _data = fh.read()

    #files = {'body': open('tests/clio4.jpg','rb')}

    start_time = time.time()
    res = requests.post(
            'http://localhost:3000/predict',
            #'https://t7p0v4a94l.execute-api.eu-west-1.amazonaws.com/Prod/post',
            #'https://4wp23cimqk.execute-api.eu-west-1.amazonaws.com/Prod/predict',
            _data)
            #files=files)
    print(res.text)
    print("{:.3f}".format(time.time() - start_time))


async def main():
    mydata = list()
    filepath = 'clio-peugeot.json'
    with open(filepath) as fh:
        mydata.append(fh.read())

    filepath = 'clio4.json'
    with open(filepath) as fh:
        mydata.append(fh.read())

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_test) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor, calling, mydata[i%2]
            )
            for i in range(num_test)
        ]
        start_time = time.time()
        print('Starting')
        for response in await asyncio.gather(*futures):
            #print("{:.3f}".format(time.time() - start_time))
            pass
        print("End {:.3f} - {:.3f}".format(
            time.time() - start_time,
            (time.time() - start_time)/num_test))

#loop = asyncio.get_event_loop()
#loop.run_until_complete(main())
single_call()
