CC = mpicc
CFLAGS = -std=c11 -O3

SRCDIR = .
INCDIR = ./inc
OBJDIR = obj
BINDIR = bin
MATRIXDIR = ./matrix
INSRCDIR = ./src

SRCDIRS = $(SRCDIR) $(MATRIXDIR) $(INSRCDIR)

SOURCES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.c))
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))
EXECUTABLE = $(BINDIR)/Spmm_TaRe

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) -c -o $@ $<

$(OBJDIR)/%.o: $(MATRIXDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) -c -o $@ $<

$(OBJDIR)/%.o: $(INSRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)

