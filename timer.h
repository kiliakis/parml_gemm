#ifndef __TIMER_H__
#define __TIMER_H__

#include "decls.h"
#include <sys/time.h>

#define USEC_PER_SEC 1000000L

struct xtimer {
  struct timeval elapsed_time;
  struct timeval timestamp;
};

typedef struct xtimer xtimer_t;

__BEGIN_C_DECLS
void timer_clear(xtimer_t *timer);
void timer_start(xtimer_t *timer);
void timer_stop(xtimer_t *timer);
double timer_elapsed_time(xtimer_t *timer);
__END_C_DECLS

#endif /* __TIMER_H__ */
