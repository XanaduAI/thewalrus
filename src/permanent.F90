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
    use kinds
    use vars
    use omp_lib
    implicit none

    contains

    subroutine re(mat, permanent)
        real(dp), intent(in)    :: mat(:, :)
        real(dp), intent(out)   :: permanent

        ! local variables
        real(dp), dimension(:), allocatable :: tmp
        real(dp)    :: permtmp, rowsumprod
        integer(ip) :: C, k, kg1, kg2, cntr, nmaxthreads, &
                                     j, pos, sig, sgntmp, i, ii, nthreads, n

        integer(ip), allocatable :: threadbound_low(:), threadbound_hi(:)
        real(dp), allocatable    :: tot(:), chitmp(:)

        !f2py intent(in) :: mat
        !f2py intent(out) :: permanent

#ifdef _OPENMP
        nthreads = OMP_get_max_threads()
        call omp_set_num_threads(nthreads)
#else
        nthreads = 1
#endif

        nmaxthreads = nthreads

        n = nint(sqrt(real(size(mat), dp)))

        C = 2**n-1

        nmaxthreads = nthreads

        allocate(tot(1:nmaxthreads),threadbound_low(1:nmaxthreads),threadbound_hi(1:nmaxthreads),chitmp(1:n+1), tmp(1:n))

        do i=1,nmaxthreads
            threadbound_low(i) = int(C/nmaxthreads)*(i-1)+1
            threadbound_hi(i) = int(C/nmaxthreads)*i
        end do
            threadbound_hi(nmaxthreads) =  C


        tot=0.0_dp

!$OMP PARALLEL DO private(ii,j,k,rowsumprod,kg2,sgntmp,sig,pos,tmp,permtmp,chitmp,cntr) shared(mat,tot)
        do ii = 1,nmaxthreads


            permtmp = 0.0_dp
            tmp = 0.0_dp
            chitmp = 0
            cntr = 0
            kg1 = 0
            do k = threadbound_low(ii), threadbound_hi(ii)
                rowsumprod = 1.0_dp
                kg2 = igray(k, 1)
                sgntmp = kg2-igray(k-1, 1)
                sig = sign(1, sgntmp)! sgntmp/abs(sgntmp)
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
        deallocate(tot,threadbound_low,threadbound_hi,chitmp,tmp)
    end subroutine re

    subroutine comp(mat, permanent)
        complex(dp), intent(in)   :: mat(:, :)
        complex(dp), intent(out) :: permanent

        ! local variables
        complex(dp) :: permtmp, rowsumprod
        integer(ip) :: C, k, kg1, kg2, cntr, nmaxthreads, &
                                     j, pos, sig, sgntmp, i, ii, nthreads, n

        real(dp), allocatable    :: chitmp(:)
        complex(dp), allocatable :: tot(:), tmp(:)
        integer(ip), allocatable :: threadbound_low(:), threadbound_hi(:)

        !f2py intent(in) :: mat
        !f2py intent(out) :: permanent

#ifdef _OPENMP
        nthreads = OMP_get_max_threads()
        call omp_set_num_threads(nthreads)
#else
        nthreads = 1
#endif

        nmaxthreads = nthreads

        n = nint(sqrt(real(size(mat), dp)))

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
                kg2 = igray(k, 1)
                sgntmp = kg2-igray(k-1, 1)
                sig = sign(1, sgntmp)! sgntmp/abs(sgntmp)
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
        deallocate(tot,threadbound_low,threadbound_hi,chitmp,tmp)
    end subroutine comp

    subroutine dec2bin (kk, nnn, mat)
        integer(ip), intent(in) :: kk, nnn
        real(dp), intent(out)   :: mat(1:nnn+1)

        ! local variables
        integer(ip) :: i, k

        mat(:) = 0.0_dp

        k = kk
        i = nnn

        do while (k > 0 .and. i>0)
            mat(i) = mod(k, 2)
            k = k/2 !floor(real(k, dp)/2.0_dp)
            i = i-1
        end do

        mat(nnn+1) = sum(mat(1:nnn))
    end subroutine dec2bin


    function igray(n,is)
        integer(ip), intent(in) :: n, is
        integer(ip) :: igray

        ! local variables
        integer(ip) :: idiv,ish

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
