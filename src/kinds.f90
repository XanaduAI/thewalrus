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

! Module containing custom precision types
module kinds
    implicit none

    integer, parameter :: sp = selected_real_kind(5,30)
    integer, parameter :: dp = selected_real_kind(9,99)
    integer, parameter :: qp = selected_real_kind(20, 199)
    integer, parameter :: wp = qp
    integer, parameter :: ip = selected_int_kind(16)

    ! These empty arrays are used to initialize variables to either the min or
    ! max possible number of kind wp or integer.
    real(wp), dimension(2:1)    :: empty
    integer(ip), dimension(2:1) :: iempty
end module kinds
