! Copyright 2019 Xanadu Quantum Technologies Inc.

! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at

!     http://www.apache.org/licenses/LICENSE-2.0

! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
program permanent
	use perm
	implicit none

	integer, parameter :: nmax = 10;
	integer :: n, m

	real(8) :: p
	real(8), allocatable :: mat(:, :)

	do m = 1, nmax
		! create a 2m*2m all ones matrix
		allocate(mat(2*m, 2*m))
		mat = 1.d0
		! calculate the permanent
		call re(mat, p)
		! print out the result
		write(*,*)p
		deallocate(mat)
	end do
end program
