# =============================================================================
# program.tcl -- download the bitstream without opening the Vivado GUI.
#
#   vivado -mode batch -source scripts/program.tcl -tclargs detect    # only probe
#   vivado -mode batch -source scripts/program.tcl                    # program
#   vivado -mode batch -source scripts/program.tcl -tclargs program report/board_top.bit
#   vivado -mode batch -source scripts/program.tcl -tclargs flash     # QSPI, survives power-off
#
# Structure follows Sonic-Speed-Parallelism/scripts/vivado/program.tcl (same
# board, same tooling), trimmed to what this project needs.
#
# This is a VOLATILE download: it writes the FPGA's SRAM configuration, so the
# design disappears at power-off.  That is what you want during bring-up --
# every new bitstream is one command away.  Use `flash` only when the design is
# final.
# =============================================================================
set ROOT [file normalize [file join [file dirname [info script]] ..]]
set BIT  [file join $ROOT report board_top.bit]

set MODE "program"
if {$argc > 0} {
    set a0 [lindex $argv 0]
    if {$a0 eq "detect"} {
        set MODE "detect"
    } elseif {$a0 eq "flash"} {
        set MODE "flash"
        if {$argc > 1} { set BIT [lindex $argv 1] }
    } elseif {$a0 eq "program"} {
        if {$argc > 1} { set BIT [lindex $argv 1] }
    } else {
        set BIT [file normalize $a0]
    }
}

puts "============================================================="
puts " mode     : $MODE"
if {$MODE ne "detect"} { puts " bitstream: $BIT" }
puts "============================================================="

if {$MODE ne "detect" && ![file exists $BIT]} {
    puts "❌ bitstream not found: $BIT"
    puts "   build it first:  vivado -mode batch -source scripts/impl.tcl"
    exit 1
}

open_hw_manager
if {[catch {connect_hw_server} err]} {
    puts "❌ cannot reach hw_server: $err"
    puts "   start it manually: ~/Xilinx/Vivado/2020.2/bin/hw_server &"
    exit 1
}

if {[llength [get_hw_targets -quiet]] == 0} {
    puts ""
    puts "❌ no JTAG target found.  Check, in this order:"
    puts "   1) board powered? USB-C in the JTAG port (not the UART one)?"
    puts "   2) Linux permission: a plain user needs a udev rule, or run:"
    puts "        sudo hw_server &"
    puts "   3) lsusb should list a Xilinx/Digilent/FTDI device:"
    catch {exec lsusb | grep -iE "xilinx|digilent|ftdi|0403|03fd"} _ls
    if {[info exists _ls] && $_ls ne ""} { puts "      $_ls" } \
    else { puts "      (nothing -- cable or power problem)" }
    close_hw_manager
    exit 1
}
open_hw_target

set _devs [get_hw_devices -quiet]
if {[llength $_devs] == 0} {
    puts "❌ target opened but no device on the chain."
    close_hw_manager
    exit 1
}

set DEV [lindex $_devs 0]
current_hw_device $DEV
refresh_hw_device $DEV

puts ""
puts " JTAG chain:"
foreach d $_devs {
    puts [format "   %-26s IDCODE=0x%08X PART=%s" $d \
              [get_property IDCODE $d] [get_property PART $d]]
}
set part [get_property PART $DEV]
if {[string match -nocase "*xc7a100t*" $part]} {
    puts " ✅ target device xc7a100t"
} else {
    puts " ⚠️  first device is $part, not xc7a100t -- proceeding anyway"
}

if {$MODE eq "detect"} {
    puts ""
    puts " probe only, nothing was programmed."
    close_hw_manager
    exit 0
}

# ---------------------------------------------------------------------------
# program the FPGA's configuration SRAM
# ---------------------------------------------------------------------------
set_property PROGRAM.FILE $BIT $DEV
if {[catch {program_hw_devices $DEV} err]} {
    puts "❌ program_hw_devices failed: $err"
    close_hw_manager
    exit 1
}
refresh_hw_device -update_hw_probes false $DEV
puts ""
puts " ✅ programmed (volatile -- gone at power-off)"

# ---------------------------------------------------------------------------
# QSPI flash (optional, survives power-off)
#
# The board carries a **W25Q128JVSQ**, not a Spansion s25fl128s.  Picking the
# wrong cfgmem part fails with
#     "Flash Programming Unsuccessful: Failure to set flash parameters"
# which looks like a configuration error but is really "wrong chip model", so
# candidates are tried in the order that is known to work on this board.
#
# The same error also appears if the FPGA is not currently running a design with
# a STARTUP/SPI core, because program_hw_cfgmem reaches the flash *through* it.
# Hence: create the cfgmem object, then (re)program the SRAM, then erase/write.
# ---------------------------------------------------------------------------
if {$MODE eq "flash"} {
    set BIN [file rootname $BIT].bin

    puts ""
    puts "  [1/4] write_cfgmem (QSPI x4, 16 MB)"
    if {[catch {write_cfgmem -force -format bin -interface SPIx4 -size 16 \
                    -loadbit "up 0x0 $BIT" $BIN} err]} {
        puts "❌ write_cfgmem failed: $err"; close_hw_manager; exit 1
    }
    puts "        $BIN ([expr {[file size $BIN]/1024}] KB)"

    puts "  [2/4] attach the cfgmem object"
    set CFGMEM_PART ""
    foreach cand [list \
            "w25q128bv-spi-x1_x2_x4" \
            "mt25ql128-spi-x1_x2_x4" \
            "s25fl128sxxxxxx0-spi-x1_x2_x4"] {
        if {[llength [get_cfgmem_parts -quiet $cand]] > 0} { set CFGMEM_PART $cand; break }
    }
    if {$CFGMEM_PART eq ""} {
        puts "  ⚠️  no 128 Mb QSPI candidate in the local part library;"
        puts "      check the flash chip on your board before trusting this."
        set CFGMEM_PART [lindex [get_cfgmem_parts] 0]
    }
    puts "        using $CFGMEM_PART"
    # on failure the next thing you want is the list of what *is* available
    catch {
        set _seen {}
        foreach pp [get_cfgmem_parts] {
            if {[string match -nocase "*w25q*" $pp] || [string match -nocase "*128*qspi*" $pp]} {
                if {[lsearch -exact $_seen $pp] < 0} { lappend _seen $pp }
            }
        }
        if {[llength $_seen] > 0} {
            puts "        128Mb QSPI candidates in the local library:"
            foreach pp $_seen { puts "          - $pp" }
        }
    }

    if {[llength [get_property PROGRAM.HW_CFGMEM $DEV -quiet]] == 0} {
        create_hw_cfgmem -hw_device $DEV -mem_dev [lindex [get_cfgmem_parts $CFGMEM_PART] 0]
    }
    set CFG [get_property PROGRAM.HW_CFGMEM $DEV]

    set_property PROGRAM.BLANK_CHECK   0 $CFG
    set_property PROGRAM.ERASE         1 $CFG
    set_property PROGRAM.CFG_PROGRAM   1 $CFG
    set_property PROGRAM.VERIFY        1 $CFG
    set_property PROGRAM.CHECKSUM      0 $CFG
    set_property PROGRAM.ADDRESS_RANGE {use_file} $CFG
    set_property PROGRAM.FILES         [list $BIN] $CFG
    set_property PROGRAM.PRM_FILE      {} $CFG
    set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $CFG

    puts "  [3/4] program SRAM first (this is the channel to the flash)"
    set BITS [get_property PROGRAM.HW_CFGMEM_BITFILE $DEV]
    if {$BITS eq ""} { set BITS $BIT }
    create_hw_bitstream -hw_device $DEV $BITS
    if {[catch {program_hw_devices $DEV} err]} {
        puts "❌ SRAM program failed: $err"; close_hw_manager; exit 1
    }
    refresh_hw_device $DEV

    puts "  [4/4] erase + write + verify (1-3 minutes, do not unplug)"
    if {[catch {program_hw_cfgmem -hw_cfgmem $CFG} err]} {
        puts "❌ flash write failed: $err"; close_hw_manager; exit 1
    }
    puts ""
    puts " ✅ QSPI flash written and verified."
    puts "    Power-cycle to boot from flash.  Note that JTAG wins over QSPI:"
    puts "    unplug the JTAG cable before testing the standalone boot."
}

close_hw_manager
puts "============================================================="
puts " done."
puts "============================================================="
