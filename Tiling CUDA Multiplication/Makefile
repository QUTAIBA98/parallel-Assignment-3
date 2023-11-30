
all: TilingMult


TilingMult: tiling_mult.cu
	nvcc -o tiling_mult tiling_mult.cu -lcudart -lcuda -lm

clean:
	$(RM) tiling_mult


run: run_TilingMult

run_TilingMult:
	./tiling_mult mat_a.csv 1000 1000 mat_b.csv 1000 1000 resultant_matrix.csv
	

