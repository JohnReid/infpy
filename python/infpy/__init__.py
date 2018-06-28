#
# Copyright John Reid 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2018
#

import pkg_resources

__doc__ = pkg_resources.resource_string(__name__, "README")
__license__ = pkg_resources.resource_string(__name__, "LICENSE")
__release__, __svn_revision__ = pkg_resources.resource_string(
    __name__, "VERSION").strip().split(b'-')
__major_version__, __minor_version__, __release_version__ = list(map(
    int, __release__.split(b'.')))
__version__ = '%d.%d' % (__major_version__, __minor_version__)


def version_string():
    """Return the release and svn revision as a string."""
    return '%s %s' % (__release__, __svn_revision__)


from .utils import *
from .distribution import *
