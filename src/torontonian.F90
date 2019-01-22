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
    integer, parameter :: qp = selected_real_kind(20,199)
    integer, parameter :: wp = dp
    integer, parameter :: ip = selected_int_kind(16)

    ! These empty arrays are used to initialize variables to either the min or
    ! max possible number of kind wp or integer.
    real(wp)    :: empty(2:1)
    integer(ip) :: iempty(2:1)
end module kinds

! Module containing some common variables
module vars
    use kinds
    implicit none

    integer(ip)         :: n, ell
    real(wp), parameter :: eps = 1.0d-6
    real(wp), parameter :: pi = acos(-1.0_wp)
end module vars

! Module containing some useful structures
module structures
    use kinds
    implicit none

    type GaussianState
        real(wp), allocatable :: rbar(:)   ! Mean vector of the state
        real(wp), allocatable :: V(:,:)    ! Covariance matrix of the state
        real(wp)              :: C         ! Probability Coefficients
    end type GaussianState

    real(wp) :: time, dtime
end module structures

module torontonian_samples
    use kinds
    use vars
    use structures

    implicit none

    contains

    subroutine GenerateSample(covmat, mean, n_sample, seed, sample_out)
        real(wp), intent(in)     :: covmat(:,:)
        real(wp), intent(in)     :: mean(:)
        integer(ip)              :: n_sample
        integer(ip), intent(in)  :: seed
        integer(ip), intent(out) :: sample_out(1:n_sample)

        ! local variables
        integer                  :: seed_tmp
        integer(ip), allocatable :: sample(:)

        integer(ip)                      :: i, j, k, ntmp, jj, NewStateSize
        real(wp)                         :: qq, qq1, qq2, random, TotalProb
        type(GaussianState)              :: state(1)
        type(GaussianState), allocatable :: state_tmp1(:), state_tmp2(:)

        !f2py intent(in) :: covmat
        !f2py intent(in) :: mean, seed, n_sample
        !f2py intent(out) :: sample_out

        seed_tmp = seed
        n = size(mean)
        ell = n/2

        call random_seed(seed_tmp)

        allocate(sample(1:ell))

        do i=1,size(state)
            allocate(state(i)%rbar(1:n),state(i)%V(1:n,1:n))
        end do

        state(1)%rbar = mean
        state(1)%V = covmat

        allocate(state_tmp1(1:size(state)))

        state_tmp1 = state

        call compute_q(state_tmp1(1)%V, state_tmp1(1)%rbar, qq)

        TotalProb = qq

        state_tmp1(:)%C = 1.0_wp

        do i=1,ell

            ntmp = size(state_tmp1)

            call random_number(random)

            NewStateSize = size(state_tmp1(1)%rbar) - 2

            if (NewStateSize >= 2) then
                if (qq > random) then

                    sample(i) = 0

                    do k=1,size(state_tmp1)
                        call compute_q(state_tmp1(k)%v, state_tmp1(k)%rbar, qq1)
                        state_tmp1(k)%c = state_tmp1(k)%c * qq1/qq
                    end do

                    allocate(state_tmp2(1:ntmp))

                    do jj=1,ntmp
                        allocate(state_tmp2(jj)%rbar(1:NewStateSize), &
                                 state_tmp2(jj)%V(1:NewStateSize, 1:NewStateSize))
                    end do

                    do j=1,ntmp
                        state_tmp2(j)%V = NewCovMat(state_tmp1(j)%V)
                        state_tmp2(j)%rbar = NewMean(state_tmp1(j)%rbar, state_tmp1(j)%V)
                        state_tmp2(j)%C = state_tmp1(j)%C
                    end do

                    deallocate(state_tmp1)
                    allocate(state_tmp1(1:size(state_tmp2)))
                    state_tmp1 = state_tmp2
                    deallocate(state_tmp2)

                    qq2 = 0.0_wp
                    qq1 = 0.0_wp

                    do k=1,size(state_tmp1)
                        call compute_q(state_tmp1(k)%v, state_tmp1(k)%rbar, qq1)
                        qq2 = qq2 + qq1*state_tmp1(k)%c
                    end do

                else if (qq <= random) then

                    sample(i) = 1
                        allocate(state_tmp2(1:2*ntmp))
                        do jj=1,2*ntmp
                            allocate(state_tmp2(jj)%rbar(1:NewStateSize), &
                                     state_tmp2(jj)%V(1:NewStateSize, 1:NewStateSize))
                        end do

                        do j=1,ntmp
                            state_tmp2(2*j-1)%V = NewCovMat(state_tmp1(j)%V)
                            state_tmp2(2*j-1)%rbar = NewMean(state_tmp1(j)%rbar, state_tmp1(j)%V)
                            call compute_q(state_tmp1(j)%V, state_tmp1(j)%rbar, qq1)
                            state_tmp2(2*j-1)%C = - state_tmp1(j)%C * qq1/(1.0_wp-qq)

                            state_tmp2(2*j)%V = VA(state_tmp1(j)%V)
                            state_tmp2(2*j)%rbar = rA(state_tmp1(j)%rbar, state_tmp1(j)%V)
                            state_tmp2(2*j)%C = state_tmp1(j)%C /(1.0_wp-qq)
                        end do

                    deallocate(state_tmp1)
                    allocate(state_tmp1(1:size(state_tmp2)))

                    state_tmp1 = state_tmp2

                    deallocate(state_tmp2)

                    qq2 = 0.0_wp
                    qq1 = 0.0_wp

                    if (abs(sum(state_tmp1(:)%C) - 1.0_wp) > 1e-6) then
                        write(*, *)"Error: Sum of all coefficients not 1. Sum =", sum(state_tmp1(:)%C)
                        STOP
                    end if

                    do k=1,size(state_tmp1)
                        call compute_q(state_tmp1(k)%V, state_tmp1(k)%rbar, qq1)
                        qq2 = qq2 + qq1*state_tmp1(k)%C
                    end do

                end if

            else
                if (qq > random) then
                     sample(i) = 0
                else if (qq <= random) then
                     sample(i) = 1
                end if
            end if

            qq = qq2

            if (qq < -eps) then
                print*, "Error: Probability is less than 0. q = ", qq
            else if (qq - 1.0_wp > 0.01_wp) then
                print*, "Error: Probability is less greater than 1. q = ", qq
            end if

            if (qq < 0.0_wp) then
                print*, 'Warning: probability was a small negative number, q=', qq, "Set to Zero!"
                qq = 0.0_wp
            end if

            TotalProb = TotalProb * qq

        end do

        forall(i=1:ell) sample_out(i) = sample(ell+1-i)
    end subroutine GenerateSample

    pure function VA(mat)
        real(wp), intent(in) :: mat(:, :)
        real(wp), allocatable :: VA(:, :)

        ! local variables
        integer(ip) :: i, nn

        nn = sqrt(real(size(mat),wp))

        allocate(VA(1:nn-2,1:nn-2))

        VA = mat(1:nn-2,1:nn-2)
    end function VA

    pure function rA(rbar, mat)
        real(wp), intent(in)  :: rbar(:)
        real(wp), intent(in)  :: mat(:, :)
        real(wp), allocatable :: rA(:)

        ! local variables
        integer(ip) :: i, nn

        nn = sqrt(real(size(mat),wp))

        allocate(rA(1:nn-2) )

        rA = rbar(1:nn-2)
    end function rA

    pure function NewMean(rbar, mat)
        real(wp), intent(in)  :: rbar(:)
        real(wp), intent(in)  :: mat(:, :)
        real(wp), allocatable :: NewMean(:)

        ! local variables
        real(wp), allocatable  :: rA(:),  VAB(:,:)
        real(wp) :: rB(1:2), VB(2,2), VB_inv(2,2), detVB
        integer(ip) :: i, nn, IdSize

        IdSize = 2

        nn = sqrt(real(size(mat),wp))

        allocate(VAB(1:nn-2,1:2), NewMean(1:nn-2), rA(1:nn-2) )

        rB = rbar(nn-1:nn)
        rA = rbar(1:nn-2)

        VAB = mat(1:nn-2,nn-1:nn)
        VB = mat(nn-1:nn,nn-1:nn) + identity(IdSize)
        detVB = VB(1,1)*VB(2,2) - VB(1,2)*VB(2,1)
        VB_inv = reshape((/VB(2,2), -VB(2,1), -VB(1,2), VB(1,1)/), (/2,2/))/detVB

        NewMean(:) = rA - matmul(VAB, matmul(VB_inv,rB))
    end function NewMean

    pure function NewCovMat(mat)
        real(wp), intent(in) :: mat(:,:)
        real(wp), allocatable :: NewCovMat(:,:)

        ! local variables
        real(wp) :: VB(2,2), VB_inv(2,2), detVB
        real(wp), allocatable :: VAB(:,:), VAB_T(:,:), VA(:,:)
        integer(ip) :: i, nn, IdSize

        IdSize = 2

        nn = sqrt(real(size(mat),wp))

        allocate(VAB(1:nn-2,1:2), VAB_T(1:2,1:nn-2), VA(1:nn-2,1:nn-2),NewCovMat(1:nn-2,1:nn-2))

        VA = mat(1:nn-2,1:nn-2)

        VAB = mat(1:nn-2,nn-1:nn)
        VAB_T = mat(nn-1:nn,1:nn-2)

        VB = mat(nn-1:nn,nn-1:nn) + identity(IdSize)
        detVB = VB(1,1)*VB(2,2) - VB(1,2)*VB(2,1)
        VB_inv = reshape((/VB(2,2), -VB(2,1), -VB(1,2), VB(1,1)/), (/2,2/))/detVB

        NewCovMat(:,:) = VA - matmul(VAB, matmul(VB_inv,VAB_T))

        deallocate(VAB, VAB_T, VA)
    end function NewCovMat

    subroutine Compute_q(mat, rbar, q)
        real(wp), intent(in) :: mat(:,:)
        real(wp), intent(in) :: rbar(:)
        real(wp), intent(out) :: q

        ! local variables
        real(wp)    :: VB(2,2), VB_inv(2,2), rbar_B_T(1,1:2), detVB, detVB_inv, tmp(1)
        integer(ip) :: i, nn, IdSize

        real(wp), allocatable :: rbar_B(:)

        nn = sqrt(real(size(mat),wp))

        IdSize = 2

        allocate(rbar_B(1:2))

        rbar_B(1:2) = rbar(nn-1:nn)
        rbar_B_T(1,1:2) = rbar(nn-1:nn)

        VB = mat(nn-1:nn,nn-1:nn) + identity(IdSize)
        detVB = VB(1,1)*VB(2,2) - VB(1,2)*VB(2,1)

        VB_inv = reshape((/VB(2,2), -VB(2,1), -VB(1,2), VB(1,1)/), (/2,2/))/detVB
        detVB_inv = VB_inv(1,1)*VB_inv(2,2) - VB_inv(1,2)*VB_inv(2,1)

        tmp = -matmul(rbar_B_T,matmul(VB_inv, rbar_B))

        q = 2.0_wp*exp(tmp(1))/sqrt(detVB)
    end subroutine Compute_q

    pure function identity(nn)
        integer(ip), intent(in) :: nn
        real(wp) :: identity(1:nn,1:nn)

        ! local variables
        integer(ip) :: i, j

        forall(i=1:nn, j=1:nn) identity(i,j) = (i/j)*(j/i)
    end function identity

end module torontonian_samples
