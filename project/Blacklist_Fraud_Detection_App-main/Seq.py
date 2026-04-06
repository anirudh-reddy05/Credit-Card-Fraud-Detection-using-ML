# Shim to expose XBNet.Seq as top-level Seq for unpickling compatibility
from XBNet.Seq import Seq
__all__ = ['Seq']
