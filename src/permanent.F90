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

module perm
    ! Module to compute the permanent of a matrix based on Ryser's formula with Gray code implementation.
    ! Subroutines 're' and 'comp' are respectively for real and complex matrices. 
    ! The algorithm has the complexity n*2^n with Gray code implementation. We
    ! use OpenMP parallelization to take advantage of share memory parallelism. When
    ! implemented in double precision, numerical errors become as large as 100% for a
    ! 30x30 matrix. This stems from summation of ~2^30 double precision numbers. In
    ! order to avoid this, we have also implemented the algorithm in quad precision
    ! which can be availed by changing wp = qp from wp = dp in kinds.f90 and compiling
    ! the program again. 

    use kinds
    use vars
    use omp_lib
    implicit none

    contains

    subroutine re(matin, permanent_out)
        ! Computing permanent of a real matrix
        real(dp), intent(in)    :: matin(:, :)
        real(dp), intent(out)   :: permanent_out

        real(wp), allocatable    :: mat(:, :)
        real(wp)   :: permanent

        ! local variables
        real(wp), dimension(:), allocatable :: tmp
        real(wp)    :: permtmp, rowsumprod
        integer(ip2) :: C, k, kg1, kg2, cntr, &
                                     j, pos, sig, sgntmp, i, n

        integer(ip) :: nmaxthreads, nthreads, ii

        integer(ip2), allocatable :: threadbound_low(:), threadbound_hi(:)
        real(wp), allocatable    :: tot(:)
        real(dp), allocatable    :: chitmp(:)

        !f2py intent(in) :: matin
        !f2py intent(out) :: permanent_out

#ifdef _OPENMP
        nthreads = OMP_get_max_threads()
        call omp_set_num_threads(nthreads)
#else
        nthreads = 1
#endif

        n = nint(sqrt(real(size(matin), wp)))
        allocate(mat(1:n, 1:n))

        mat = real(matin, wp)
        nmaxthreads = nthreads


        C = 2**n-1

        nmaxthreads = nthreads


        allocate(tot(1:nmaxthreads),threadbound_low(1:nmaxthreads),threadbound_hi(1:nmaxthreads),chitmp(1:n+1), tmp(1:n))

        do i=1,nmaxthreads
            threadbound_low(i) = int(C/nmaxthreads)*(i-1)+1
            threadbound_hi(i) = int(C/nmaxthreads)*i
        end do
            threadbound_hi(nmaxthreads) =  C


        tot=0.0_wp

!$OMP PARALLEL DO private(ii,j,k,rowsumprod,kg2,sgntmp,sig,pos,tmp,permtmp,chitmp,cntr) shared(mat,tot)
        do ii = 1,nmaxthreads


            permtmp = 0.0_wp
            tmp = 0.0_wp
            chitmp = 0
            cntr = 0
            kg1 = 0

            do k = threadbound_low(ii), threadbound_hi(ii)
                rowsumprod = 1.0_wp
                kg2 = igray(k, 1_ip2)
                sgntmp = kg2-igray(k-1_ip2, 1_ip2)
                sig = sign(1_ip2, sgntmp)! sgntmp/abs(sgntmp)
                pos = 0

                do while(ibits(sgntmp, pos, 1) < 1)
                     pos = pos+1
                end do
                pos = n-pos


                if (k == threadbound_low(ii)) then
                    call dec2bin(kg2, n, chitmp)

                    do j=1,n
                        tmp(j) = tmp(j) + sum(chitmp(1:n) * mat(1:n,j))
                        rowsumprod = rowsumprod * tmp(j)
                    end do

                    cntr = int(chitmp(n+1), ip)

                else

                    cntr = cntr+sig

                    do j=1,n

                        if (sig < 0) then
                            tmp(j) = tmp(j) - mat(pos, j)!sign(mat(pos, j), sgntmp)
                        else
                            tmp(j) = tmp(j) + mat(pos, j)!sign(mat(pos, j), sgntmp)
                        end if

                        rowsumprod = rowsumprod * tmp(j)
                    end do

                end if


                if(mod(n-cntr,2)==0) then
                    permtmp = permtmp + rowsumprod
                else
                    permtmp = permtmp - rowsumprod
                end if

            end do

            tot(ii) = permtmp

        end do
!$OMP END PARALLEL DO

        permanent = sum(tot)
        permanent_out = real(permanent, dp)

        deallocate(tot,threadbound_low,threadbound_hi,chitmp,tmp)
    end subroutine re

    subroutine comp(matin, permanent_out)
        ! Computing permanent of a complex matrix
        complex(dp), intent(in)   :: matin(:, :)
        complex(dp), intent(out) :: permanent_out  

        complex(wp), allocatable :: mat(:, :)
        complex(wp) :: permanent

        ! local variables
        complex(wp) :: permtmp, rowsumprod
        integer(ip2) :: C, k, kg1, kg2, cntr, &
                                     j, pos, sig, sgntmp, i, n

        integer(ip) :: nmaxthreads, nthreads, ii

        integer(ip2), allocatable :: threadbound_low(:), threadbound_hi(:)


        real(dp), allocatable    :: chitmp(:)
        complex(wp), allocatable :: tot(:), tmp(:)

        !f2py intent(in) :: mat
        !f2py intent(out) :: permanent_out

#ifdef _OPENMP
        nthreads = OMP_get_max_threads()
        call omp_set_num_threads(nthreads)
#else
        nthreads = 1
#endif

        n = nint(sqrt(real(size(matin), wp)))
        allocate(mat(1:n, 1:n))

        mat = real(matin, wp) + zi*real(aimag(matin), wp)

        nmaxthreads = nthreads

        C = 2**n-1

        nmaxthreads = nthreads

        allocate(tot(1:nmaxthreads), threadbound_low(1:nmaxthreads), &
            threadbound_hi(1:nmaxthreads), chitmp(1:n+1), tmp(1:n))

        do i=1,nmaxthreads
            threadbound_low(i) = int(C/nmaxthreads)*(i-1)+1
            threadbound_hi(i) = int(C/nmaxthreads)*i
        end do
            threadbound_hi(nmaxthreads) = C

        tot = zzero
        chitmp = 0

!$OMP PARALLEL DO private(ii,j,k,rowsumprod,kg2,sgntmp,sig,pos,tmp,permtmp,chitmp,cntr) shared(mat,tot)
        do ii=1, nmaxthreads

            permtmp = zzero
            tmp = zzero
            cntr = 0
            kg1 = 0
            do k=threadbound_low(ii), threadbound_hi(ii)

                rowsumprod = zone
                kg2 = igray(k, 1_ip2)
                sgntmp = kg2-igray(k-1_ip2, 1_ip2)
                sig = sign(1_ip2, sgntmp)! sgntmp/abs(sgntmp)
                pos = 0

                do while(ibits(sgntmp, pos, 1) < 1)
                     pos = pos+1
                end do
                pos = n-pos


                if (k == threadbound_low(ii)) then
                    call dec2bin(kg2, n, chitmp)

                    do j=1, n
                        tmp(j) = tmp(j) + sum(chitmp(1:n) * mat(1:n,j))

                        rowsumprod = rowsumprod * tmp(j)
                    end do

                    cntr = int(chitmp(n+1), ip)

                else

                    cntr = cntr+sig

                    do j=1,n

                        if (sig < 0) then
                            tmp(j) = tmp(j) - mat(pos, j)!sign(mat(pos, j), sgntmp)
                        else
                            tmp(j) = tmp(j) + mat(pos, j)!sign(mat(pos, j), sgntmp)
                        end if

                        rowsumprod = rowsumprod * tmp(j)
                    end do

                end if



                if(mod(n-cntr,2)==0) then
                    permtmp = permtmp + rowsumprod
                else
                    permtmp = permtmp - rowsumprod
                end if

            end do

            tot(ii) = permtmp

        end do
!$OMP END PARALLEL DO

        permanent = sum(tot)
        permanent_out = real(permanent, dp) + (0.0_dp, 1.0_dp)*real(aimag(permanent), dp)

        deallocate(tot,threadbound_low,threadbound_hi,chitmp,tmp)
    end subroutine comp

    subroutine dec2bin (kk, nnn, mat)
        integer(ip2), intent(in) :: kk, nnn
        real(dp), intent(out)   :: mat(1:nnn+1)

        ! local variables
        integer(ip2) :: i, k

        mat(:) = 0.0_dp

        k = kk
        i = nnn

        do while (k > 0 .and. i>0)
            mat(i) = mod(k, 2)
            k = k/2 !floor(real(k, wp)/2.0_wp)
            i = i-1
        end do

        mat(nnn+1) = sum(mat(1:nnn))
    end subroutine dec2bin


    function igray(n,is)
        integer(ip2), intent(in) :: n, is
        integer(ip2) :: igray

        ! local variables
        integer(ip2) :: idiv,ish

        if (is >= 0) then
                igray = ieor(n,n/2)
        else
                ish = -1
                igray = n
                do
                    idiv = ishft(igray,ish)
                    igray = ieor(igray,idiv)
                    if (idiv <= 1 .or. ish == -16) RETURN
                        ish = ish+ish
                end do
        end if
    end function igray

end module perm
