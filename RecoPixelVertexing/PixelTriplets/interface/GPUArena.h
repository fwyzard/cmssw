#ifndef GPUARENA_H_
#define GPUARENA_H_
#include "cuda_runtime.h"


template <int CHUNK_SIZE, typename T>
class GPUChunk {
	//To think about: maybe it would be better to move the
	//nextFreeValue and the next pointer out of the chunk
	//and make their own lists inside the arena
	//if a chunk only consists of CHUNK_SIZE values then each
	//thread can load one memory transaction worth of pure data
	//whereas the nextFreeValue and next pointers ruin that
    public:
	T values[CHUNK_SIZE];
	int nextFreeValue;
	GPUChunk<CHUNK_SIZE, T> *next;

	__device__ int num_values_in_chunk() {
		return nextFreeValue;
	}

	__device__ bool push_back(T value) {
		int id = atomicAdd(&nextFreeValue, 1);
		if(id < CHUNK_SIZE) {
			//found space
			values[id] = value;
			return true;
		} else {
			//chunk is full and this thread must get a new one
			return false;
		}
	}

	__device__ T& get_element_at(int i) {
		return values[i];
	}
};

//This iterator starts at the head chunk for a given element on a given layer
//and iterates "backwards"
template <int CHUNK_SIZE,typename T>
class GPUArenaIterator {
    private:
    GPUChunk<CHUNK_SIZE, T> *currentChunk;
    int cursorInChunk;

    public:
    __device__ GPUArenaIterator(GPUChunk<CHUNK_SIZE, T> *head_chunk) {
        currentChunk = head_chunk;
        if(currentChunk != NULL) {
            cursorInChunk = currentChunk->num_values_in_chunk() - 1;
        }
    }

    __device__ bool has_next() {
        return currentChunk != NULL && (cursorInChunk >= 0 || currentChunk->next != NULL);
    }

    __device__ T& get_next() {
        if(cursorInChunk < 0) {
            //No more elements left in chunk, go to next chunk
            //assuming there are more chunks because you hopefully called hasNext before
            currentChunk = currentChunk->next;
            cursorInChunk = currentChunk->num_values_in_chunk() - 1;
        }
        return currentChunk->get_element_at(cursorInChunk--);
    }
};

template <int CHUNK_SIZE, typename T>
__global__ void init_mappings_kernel(GPUChunk<CHUNK_SIZE, T> **mappings, GPUChunk<CHUNK_SIZE, T> *startOfChunks, int offset, int numElements) {
    for(int mySlot = threadIdx.x + blockIdx.x * blockDim.x; mySlot < numElements; mySlot += gridDim.x * blockDim.x) {
        mappings[mySlot] = startOfChunks + offset + mySlot;
    }
};

template <int NumLayers, int CHUNK_SIZE, typename T>
class GPUArena {
    private:
	//how many elements does the arena store per layer
	int numElementsPerLayer[NumLayers];
	//a map from an element id (per layer) to the head of the chunk linked list that stores the values
	GPUChunk<CHUNK_SIZE, T> **mappingIdToCurrentChunk[NumLayers];
	//the shared chunks
	GPUChunk<CHUNK_SIZE, T> *chunks;
	//how many chunks are there in total
	int capacity;
	//shared cursor to indicate the next free chunk
	//next free chunk does not start out as 0 but every element in every layer
	//by default gets a chunk
	int nextFreeChunk;

    public:

	GPUArena(int _capacity)
	: capacity(_capacity)
	{
		//allocate the main arena storage and set everything to 0 (important
		//because the counters in each chunk must be )
		cudaMalloc(&chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity);
		cudaMemset(chunks, sizeof(GPUChunk<CHUNK_SIZE, T>) * capacity, 0);
		nextFreeChunk = 0;
	}

	void init_layer(int layer, int numElementsOnLayer) {
        numElementsPerLayer[layer] = numElementsOnLayer;

		//each element implicitly gets its own initial chunk

        size_t mapSizeInBytes = sizeof(GPUChunk<CHUNK_SIZE, T>*) * numElementsOnLayer;

		cudaMalloc(&mappingIdToCurrentChunk[layer], mapSizeInBytes);

        init_mappings_kernel<<<64, 16>>>(mappingIdToCurrentChunk[layer], chunks, nextFreeChunk, numElementsOnLayer);
		nextFreeChunk += numElementsPerLayer[layer];
	}

    __device__ int get_num_elements_per_layer(int layer) {
        return numElementsPerLayer[layer];
    }

    __device__ __noinline__ int get_fresh_chunk_id() {
        int id = atomicAdd(&nextFreeChunk, 1);
        return id;
    }
	__device__ GPUChunk<CHUNK_SIZE, T>* get_new_chunk() {
        int id = get_fresh_chunk_id();
        if(id >= capacity) {
			printf("PANIC: GPUArena ran out of capacity\n");
            assert(false);
            return NULL;
		}
		return &chunks[id];
	}

	__device__ GPUChunk<CHUNK_SIZE, T>* get_head_chunk(int layer, int elementId) {
		return mappingIdToCurrentChunk[layer][elementId];
	}

	__device__ GPUArenaIterator<CHUNK_SIZE, T> iterator(int layer, int elementId) {
		return GPUArenaIterator<CHUNK_SIZE, T>(get_head_chunk(layer, elementId));
	}

	__device__ void push_back(int layer, int elementId, T &value) {


		GPUChunk<CHUNK_SIZE, T> *currentChunk = get_head_chunk(layer, elementId);
		assert(currentChunk);

		while(true) {
			bool status = currentChunk->push_back(value);
			if(status == true) {
				//we were able to snatch a value spot in the chunk, done
				break;
			} else {
				//chunk is full. Every thread seeing a full chunk gets a new
				//one and tries to add it. Because the GPU doesn't guarantee
				GPUChunk<CHUNK_SIZE, T> *newChunk = get_new_chunk();
				newChunk->next = currentChunk; //hook up list
				//Note: we don't need a threadfence_system here because we are
				//either only writing or only reading, never both. And while writing
				//nobody cares about the next pointer
				currentChunk = (GPUChunk<CHUNK_SIZE, T>*)atomicCAS((unsigned long long int *)&mappingIdToCurrentChunk[layer][elementId], (unsigned long long int)currentChunk, (unsigned long long int)newChunk);
			}
		}
	}

};
//
//__global__ void testWrite(GPUArena<NUM_LAYERS, int> arena, int howMany) {
//    int myLayer =  blockIdx.y;
//    assert(myLayer < NUM_LAYERS);
//    int numElements = arena.get_num_elements_per_layer(myLayer);
//
//    for(int myself = threadIdx.x + blockIdx.x * blockDim.x; myself < howMany; myself++) {
//        //we want multiple concurrent writes to the same element
//        int myElementId = myself % numElements;
//        arena.push_back(myLayer, myElementId, myElementId);
//    }
//};
//
//__global__ void testRead(GPUArena<NUM_LAYERS, int> arena, int howMany) {
//    int myLayer =  blockIdx.y;
//    assert(myLayer < NUM_LAYERS);
//    int numElements = arena.get_num_elements_per_layer(myLayer);
//
//    int myElementId = threadIdx.x + blockIdx.x * blockDim.x;
//    if(myElementId < numElements) {
//        GPUArenaIterator<int> it = arena.iterator(myLayer, myElementId);
//        int count = 0;
//        while(it.has_next()) {
//            int value = it.get_next();
//            if(value != myElementId) {
//                printf("PANIC, list for elementId %d is wrong. Expected %d, got %d\n", myElementId, value, myElementId);
//            }
//            count++;
//        }
//        if(count != howMany / numElements) {
//            printf("PANIC, list size for elementId %d is wrong. Expected %d but found %d\n", myElementId, howMany/numElements, count);
//        }
//    }
//};
#endif
