
tserverlite -cf=tserver.cf -test=umc210.6
tserverlite -cf=tserver.cf -test=umc320.5
tserverlite -cf=tserver.cf -test=umc321.6
tserverlite -cf=tserver.cf -test=umc321.101
tserverlite -cf=tserver.cf -test=umc323.7
tserverlite -cf=tserver.cf -test=umc404.1

#./tserverlite -d=1 -test=umc1.2     -log -disable_libdce
#./tserverlite -d=1 -test=umc403.1   -log -disable_libdce
./tserverlite -cf=565.cf -test=umc56.5   -disable_libdce
./tserverlite -cf=565.cf -test=umc56.6   -disable_libdce
./tserverlite -cf=565.cf -test=umc56.25  -disable_libdce
./tserverlite -cf=565.cf -test=umc56.26  -disable_libdce

./tserverlite -cf=tserver.cf -test=std1.6 -disable_libdce
./tserverlite -cf=tserver.cf -test=std2.6 -disable_libdce
./tserverlite -cf=tserver.cf -test=std3.6 -disable_libdce
./tserverlite -cf=tserver.cf -test=std4.6 -disable_libdce

sleep 1
./tserverlite -tsl_load_ucode=false -tc_load_ucode=true -d=1 -test=PK6.40  -pm4file=gpu/pm4play/pk006040 -cf=gpu/pm4play.cf -log -disable_libdce
./tserverlite -tsl_load_ucode=false -tc_load_ucode=true -d=1 -test=PK16.40 -pm4file=gpu/pm4play/pk016040 -cf=gpu/pm4play.cf -log -disable_libdce

#./tserverlite -load_ucode=tcore -d=1 -test=PK6.40  -pm4file=gpu/pm4play/pk006040 -cf=gpu/pm4play.cf -log
#./tserverlite -load_ucode=tcore -d=1 -test=PK16.40 -pm4file=gpu/pm4play/pk016040 -cf=gpu/pm4play.cf -log


