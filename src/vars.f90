! Copyright 2019 Xanadu Quantum Technologies Inc.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

! Module containing some common variables
module vars
    use kinds
    implicit none

    ! real(wp), dimension(:, :), allocatable    :: mat, inv_mat
    !real(wp), dimension(:, :), allocatable    :: mat_comp_real, mat_comp_imag
    complex(wp), dimension(:, :), allocatable :: inv_mat_comp


    complex(wp), parameter :: zzero = (0.0_wp, 0.0_wp), zone = (1.0_wp, 0.0_wp), zi = (0.0_wp, 1.0_wp)
    real(wp), parameter :: pi = 3.141592653589793238462643383279
   
    character :: mattype*10

end module vars

