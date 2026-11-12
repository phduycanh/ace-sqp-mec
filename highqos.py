"""
Windows only: exempt the current process from power throttling (EcoQoS) and raise its priority, so that sequential
wall-clock measurements are not distorted when Windows moves a background process to slow cores or lowers its clock.
No effect on other systems.
"""
import ctypes, sys


def enable():
    if sys.platform != "win32":
        return False

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    h = ctypes.c_void_p(k.GetCurrentProcess())
    st = PROCESS_POWER_THROTTLING_STATE(1, 0x1 | 0x4, 0)   # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION: not throttled
    ok = k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))   # 4 = ProcessPowerThrottling
    k.SetPriorityClass(h, 0x80)                            # HIGH_PRIORITY_CLASS
    return bool(ok)
