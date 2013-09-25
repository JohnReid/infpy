#
# Copyright John Reid 2006
#


from test_bp import *

def cb( arg ):
    print 'In "f":', arg
callback( cb )

def test_f( arg ):
    print 'In "f":', arg, arg.value
    dir( arg )
    # arg.set_value( 'Set from python' )
    # arg.value = "Set from python"
t = test()
t.set_value( 'Set from python' )
t.set_value_via_string( 'Set from python via string' )
t.connect_slot( test_f )
