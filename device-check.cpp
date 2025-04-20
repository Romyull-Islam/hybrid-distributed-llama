#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

int main() {
    long long available_memory_mb = 0;
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    available_memory_mb = status.ullAvailPhys / (1024 * 1024);
#else
    struct sysinfo info;
    sysinfo(&info);
    available_memory_mb = info.freeram / (1024 * 1024);
#endif
    std::cout << available_memory_mb << std::endl;
    return 0;
}