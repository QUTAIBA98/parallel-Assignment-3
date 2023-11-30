
all: MatMult


MatMult: mat_mult.cu
	nvcc -o mat_mult mat_mult.cu -lcudart -lcuda -lm

clean:
	$(RM) mat_mult


run: run_MatMult

run_MatMult:
	./mat_mult mat_a.csv 1000 1000 mat_b.csv 1000 1000 resultant_matrix.csv
	

