from time import time
start = time()
from IPython.kernel import client
end = time(); print 'Took %.5s to import ipython client' % (end - start); start = end
mec = client.MultiEngineClient()
end = time(); print 'Took %.5s to create client' % (end - start); start = end
mec.push(dict(d={}, l=list()))
end = time(); print 'Took %.5s to push data' % (end - start); start = end
