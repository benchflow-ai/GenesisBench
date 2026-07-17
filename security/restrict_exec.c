#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/landlock.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef LANDLOCK_ACCESS_FS_REFER
#define LANDLOCK_ACCESS_FS_REFER (1ULL << 13)
#endif

#ifndef LANDLOCK_ACCESS_FS_TRUNCATE
#define LANDLOCK_ACCESS_FS_TRUNCATE (1ULL << 14)
#endif

static int create_ruleset(
    const struct landlock_ruleset_attr *attr,
    size_t size,
    uint32_t flags
) {
    return (int)syscall(__NR_landlock_create_ruleset, attr, size, flags);
}

static int add_rule(
    int ruleset_fd,
    enum landlock_rule_type type,
    const void *attr,
    uint32_t flags
) {
    return (int)syscall(__NR_landlock_add_rule, ruleset_fd, type, attr, flags);
}

static int restrict_self(int ruleset_fd, uint32_t flags) {
    return (int)syscall(__NR_landlock_restrict_self, ruleset_fd, flags);
}

static void die(const char *message) {
    fprintf(stderr, "restrict-exec: %s: %s\n", message, strerror(errno));
    exit(126);
}

static void allow_path(
    int ruleset_fd,
    const char *path,
    uint64_t allowed_access
) {
    int path_fd = open(path, O_PATH | O_CLOEXEC);
    if (path_fd < 0) {
        if (errno == ENOENT) {
            return;
        }
        die(path);
    }

    struct landlock_path_beneath_attr rule = {
        .allowed_access = allowed_access,
        .parent_fd = path_fd,
    };
    if (add_rule(ruleset_fd, LANDLOCK_RULE_PATH_BENEATH, &rule, 0) < 0) {
        close(path_fd);
        die("landlock_add_rule");
    }
    close(path_fd);
}

int main(int argc, char **argv) {
    if (argc < 4 || strcmp(argv[2], "--") != 0) {
        fprintf(
            stderr,
            "usage: restrict-exec RW_DIRECTORY -- COMMAND [ARG ...]\n"
        );
        return 125;
    }

    int abi = create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 1) {
        die("Landlock is unavailable");
    }

    uint64_t handled_access =
        LANDLOCK_ACCESS_FS_EXECUTE |
        LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_CHAR |
        LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG |
        LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO |
        LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_MAKE_SYM;
    if (abi >= 2) {
        handled_access |= LANDLOCK_ACCESS_FS_REFER;
    }
    if (abi >= 3) {
        handled_access |= LANDLOCK_ACCESS_FS_TRUNCATE;
    }

    struct landlock_ruleset_attr ruleset = {
        .handled_access_fs = handled_access,
    };
    int ruleset_fd = create_ruleset(&ruleset, sizeof(ruleset), 0);
    if (ruleset_fd < 0) {
        die("landlock_create_ruleset");
    }

    uint64_t read_execute =
        LANDLOCK_ACCESS_FS_EXECUTE |
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR;
    allow_path(ruleset_fd, "/usr", read_execute);
    allow_path(ruleset_fd, "/lib", read_execute);
    allow_path(ruleset_fd, "/lib64", read_execute);
    allow_path(ruleset_fd, "/opt/genesisbench", read_execute);
    allow_path(ruleset_fd, "/etc/ld.so.cache", LANDLOCK_ACCESS_FS_READ_FILE);
    allow_path(ruleset_fd, "/etc/localtime", LANDLOCK_ACCESS_FS_READ_FILE);
    allow_path(ruleset_fd, "/proc/cpuinfo", LANDLOCK_ACCESS_FS_READ_FILE);
    allow_path(ruleset_fd, "/sys/devices/system/cpu", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR);
    allow_path(
        ruleset_fd,
        "/dev/null",
        LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE
    );
    allow_path(ruleset_fd, "/dev/urandom", LANDLOCK_ACCESS_FS_READ_FILE);
    allow_path(ruleset_fd, argv[1], handled_access);

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) {
        close(ruleset_fd);
        die("PR_SET_NO_NEW_PRIVS");
    }
    if (restrict_self(ruleset_fd, 0) < 0) {
        close(ruleset_fd);
        die("landlock_restrict_self");
    }
    close(ruleset_fd);

    execvp(argv[3], &argv[3]);
    die("execvp");
    return 126;
}
