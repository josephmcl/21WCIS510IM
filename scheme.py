import sys, os, socket, time, datetime, random, string, enum

handoff = datetime.datetime(2021, 3, 14, 4, 6)


class xGridLatency(enum.Enum):
    LAtoNYC = 42
    SEAtoMIAMI = 44
    SFtoVA = 38

EXPERIMENT_LATENCY = xGridLatency.SFtoVA.value

def edge():
    '''
    '''

    port1 = 12222
    port2 = 13333
    host = '127.0.0.1'
    buf_size = 1024

    s1 = None
    s2 = None

    while datetime.datetime.now() < \
          handoff - datetime.timedelta(seconds=5):
        time.sleep(1)

    message = ''.join(random.choice(string.ascii_uppercase) for i in range(1024)).encode('utf-8')

    iters = 0
    resp = 0
    while datetime.datetime.now() < \
          handoff + datetime.timedelta(seconds=5):
        if s1 == None or s2 == None:
            try:
                s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s1.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                s1.bind((host, 9992)) 
                s1.connect((host, port1))
                s1.setblocking(0)
                s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s2.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                s2.bind((host, 9993)) 
                s2.connect((host, port2))
                s2.setblocking(0)
            except socket.error as error:
                print('[*]', error)
                if s2:
                    s2.close()
                if s1:
                    s1.close()
                sys.exit(1)

        # message = b'Hello, World.'
        # message = b'1234567890123456789012345678901234567890'
        try:
            s1.send(message)
            s2.send(message)
        except:
            pass
        iters += 1
        latency = 17 + EXPERIMENT_LATENCY
        timeout = datetime.datetime.now() + datetime.timedelta(milliseconds=latency)
        while datetime.datetime.now() < timeout:
            data = b''
            try:
                data = s1.recv(buf_size)
            except:
                pass
            if len(data) > 0: 
                print(f'[*][{datetime.datetime.now()}][1 <-> GS]', data)
                resp += 1
                break
            try:
                data = s2.recv(buf_size)
            except:
                pass
            if len(data) > 0: 
                print(f'[*][{datetime.datetime.now()}][2 <-> GS]', data)
                resp += 1
                break
    s1.close()
    s2.close()

    print('iterations', iters)
    print('messages', resp)

class satellite:
    def __init__(self, n):
        '''
        '''
        self.n = n + 1
        self.k = 1
        self.clients = {}
        self.connections = {}
        self.host = ''
        self.port = 10000 + (self.n * 1111)
        self.routes = [11111, 10000 + ((self.n + 1) * 1111)]
        if self.port > 12222: 
            self.routes.append(10000 + ((self.n - 1) * 1111))

    def up(self):
        '''
        '''
        try:
            incoming = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            incoming.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
            incoming.bind((self.host, self.port)) 
            incoming.setblocking(0)
            incoming.listen(5)

        except socket.error as error:
            if incoming:
                incoming.close()
            print('Could not open socket: ' + error)
            sys.exit(1)

        print(f'[{self.n - 1}][{datetime.datetime.now()}] Listening on {self.port}')
        while True:
            try:
                soc, address = incoming.accept()
                soc.setblocking(0)
                if address[1] < 9999:
                    time.sleep(0.01)
                else:
                    time.sleep(0.005)
                self.clients[address[1]] = (soc, address[1])
                print(f'[{self.n - 1}][{datetime.datetime.now()}] New connection from', address[1])
            except:
                pass

            for k in self.clients.keys():
                originsocket, originport = self.clients[k]
                try:
                    data = originsocket.recv(1024)
                    if len(data) == 0:
                        raise ValueError()
                except:
                    continue

                destport = self._dest(originport)

                if destport == None:
                    print(f'[{self.n - 1}][{datetime.datetime.now()}] Message from port {address[1]}' + \
                          f' orphaned.')
                    continue

                print(f'[{self.n - 1}][{datetime.datetime.now()}] Routing message to port {destport}')

                if destport in self.clients.keys() and destport not in self.routes:
                    s, _ = self.clients[destport]
                    s.send(data)
                    # s.close()
                    print(f'[{self.n - 1}][{datetime.datetime.now()}] Message from port {address[1]}' + \
                          f' forwarded to {destport}.')
                    # del clients[destport]
                    # break
                elif destport in self.routes:
                    self._route(destport, data)
                    print(f'[{self.n - 1}][{datetime.datetime.now()}] Message from port {address[1]}' + \
                          f' forwarded to {destport}.')
                else:
                    print(f'[{self.n - 1}][{datetime.datetime.now()}] Message from port {address[1]}' + \
                          f' orphaned.')
    
    def _route(self, port, data):
        '''
        '''
        host = '127.0.0.1'
        buf_size = 1024

        try:
            if port in self.connections.keys():
                s = self.connections[port]
            else:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
                s.bind((self.host, self.port + self.k)) 
                self.k += 1
            s.connect((self.host, port))
            # self.connections[port] = s 
        except socket.error as error:
            if s:
                s.close()
            print(f'[{self.n - 1}][{datetime.datetime.now()}] Could not open socket {port}:', error)
            sys.exit(1)

        s.send(data)
        s.close()
        s = None

    def _dest(self, port):
        '''
        '''
        condition = datetime.datetime.now() > handoff if self.n == 2 else datetime.datetime.now() <= handoff

        # messages from GS' are forwarded to the next satellite
        if port <= 9999:
            if condition:
                p = None
            else:
                p = 10000 + ((self.n + 1) * 1111)
        # messages from the dest. are forwarded to the prior satellite
        # unless the handoff has occured, in which case they are sent to
        # the GS. 
        elif port <= 12222 and self.n > 2:
            p = 10000 + ((self.n - 1) * 1111)
        # message from the next satellite are forwarded to the GS'
        elif port >= self.port:
            p = list(self.clients.keys())[0]
        # message from the prior satellite are forwarded to the dest.
        elif port <= self.port:
            p = 11111
        return p

class destination:

    def __init__(self):
        '''
        '''
        self.host = '127.0.0.1'
        self.k = 1
        self.port = 11111
        self.connections = {}

    def up(self):

        try:
            listening_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            listening_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
            listening_socket.bind((self.host, self.port))
            listening_socket.listen(5)
        except socket.error as error:
            if listening_socket:
                listening_socket.close()
            print('Could not open socket: ' + error)
            sys.exit(1)

        print(f'[D][{datetime.datetime.now()}] Listening on {self.port}')
        while True:
            accepted_socket, address = listening_socket.accept()
            time.sleep(0.001 * EXPERIMENT_LATENCY)
            data = accepted_socket.recv(1024)
            accepted_socket.close()
            if data:
                self._route(address[1], data)
  
    def _route(self, port, data):
        '''
        '''
        if port in self.connections.keys():
            s = self.connections[port]
        else:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((self.host, self.port + self.k)) 
                self.k += 1
                port = self._dest(port) 
                s.connect((self.host, port))
                self.connections[port] = s 
            except socket.error as error:
                if s:
                    s.close()
                print(f'[D][{datetime.datetime.now()}] Could not open socket: ', error)
                sys.exit(1)
        print(f'[D][{datetime.datetime.now()}] Message from port {self.port + self.k - 1} forwarded to {port}')
        s.send(data)
        s.close()          

    def _dest(self, port):
        '''
        '''
        if port >= 12222 and port < 13333:
            p = 12222
        elif port >= 13333 and port < 14444:
            p = 13333
        elif port < 15555:
            p = 14444
        return p

def main():
    '''
    '''
    if len(sys.argv) < 2:
        exit(-1)
    node = sys.argv[1]
    if node == 'edge':
        edge()
    elif node == 'sat1':
        satellite(1).up()
    elif node == 'sat2':
        satellite(2).up()
    elif node == 'sat3':
        satellite(3).up()
    elif node == 'dest':
        destination().up()
    else:
        print(f'Unknown option \"{node}\"')
    exit(0)

if __name__ == '__main__':
    main()