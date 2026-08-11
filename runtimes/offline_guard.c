#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <sys/socket.h>

typedef int (*socket_function)(int, int, int);
typedef int (*connect_function)(int, const struct sockaddr *, socklen_t);

static int is_network_family(int family) {
    return family == AF_INET || family == AF_INET6;
}

int socket(int domain, int type, int protocol) {
    socket_function real_socket;

    if (is_network_family(domain)) {
        errno = EPERM;
        return -1;
    }
    real_socket = (socket_function)dlsym(RTLD_NEXT, "socket");
    if (real_socket == NULL) {
        errno = ENOSYS;
        return -1;
    }
    return real_socket(domain, type, protocol);
}

int connect(int file_descriptor, const struct sockaddr *address, socklen_t address_length) {
    connect_function real_connect;

    if (address != NULL && is_network_family(address->sa_family)) {
        errno = EPERM;
        return -1;
    }
    real_connect = (connect_function)dlsym(RTLD_NEXT, "connect");
    if (real_connect == NULL) {
        errno = ENOSYS;
        return -1;
    }
    return real_connect(file_descriptor, address, address_length);
}
