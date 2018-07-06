/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "SeparateCompilationLinking/Particle/interface/v3.h"
#include "SeparateCompilationLinking/Propagate/interface/propagate.h"

int main(int argc, char ** argv)
{
	int n    = 1000000;
        int seed = 0;
	if(argc > 1)	{ n = atoi(argv[1]);}     // Number of particles
	if(argc > 2)	{ seed = atoi(argv[2]); } // Random seed

        v3 totalDistance = propagate(n, seed);

	float avgX = totalDistance.x /(float)n;
	float avgY = totalDistance.y /(float)n;
	float avgZ = totalDistance.z /(float)n;
	float avgNorm = sqrt(avgX*avgX + avgY*avgY + avgZ*avgZ);
	printf(	"Moved %d particles 100 steps. Average distance traveled is |(%f, %f, %f)| = %f\n", 
					n, avgX, avgY, avgZ, avgNorm);
	return 0;
}
