CC=mpicc
CFLAGS=-O2 -Wall

all: producer_consumer workpool

producer_consumer: producer_consumer.c
	$(CC) $(CFLAGS) -o producer_consumer producer_consumer.c

workpool: workpool.c
	$(CC) $(CFLAGS) -o workpool workpool.c

clean:
	rm -f producer_consumer workpool *.o
