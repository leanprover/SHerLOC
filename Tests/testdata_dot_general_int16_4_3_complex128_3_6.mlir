"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 0, 2], [-5, 1, 1], [4, -2, 0], [0, 0, -3]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[(-2.8660854685546662,-1.2863875102600995), (-0.70571096565419589,-1.9229616727959915), (1.7713671346768365,-0.64444997928398096), (3.1107804957108653,-1.7125102691334555), (0.95179418447559505,-1.7005646612736793), (0.70126322236172967,2.604092035591326)], [(-2.0485569678221554,-1.592446604301065), (3.2295974946601365,-1.9715341434329887), (-5.2644028617425347,-4.3836497405494042), (-2.1844284282652127,-5.0099857591004904), (-2.8728593850432218,-0.44056582660799243), (-5.0643240848858238,2.7297937308051372)], [(3.6880422346266997,2.0753398422981215), (6.5862736400961044,-4.9729379662392299), (-0.82818250048123465,1.3910207202523277), (0.79881099484328744,3.9554381704739159), (3.038540491164949,-4.5224969832724513), (6.0686213525012338,1.0619475944621799)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(4.5099990006987332,2.8642921743361436), (12.466836314538012,-11.868837605274452), (0.11500213371436718,2.1375914612206746), (4.70840248539744,6.1983660718143767), (7.0288751668054932,-10.745558627818582), (12.838505927364197,4.7279872245156858)], [(15.969912609577875,6.9148307892975538), (13.34442596302722,2.6703362543077391), (-14.949421035607953,0.22962087612282822), (-16.939519911976255,7.5080037570407026), (-4.5932898162562479,3.5397604964879532), (-2.5020188441932394,-9.2287188526893118)], [(-7.367227938574354,-1.9606568324382678), (-9.282038851937056,-3.7487784043179886), (17.614274262192417,6.1894995639628849), (16.811978839373886,3.1699304416671588), (9.5528955079888238,-5.9211269918787321), (12.933701059218567,4.9567806807550294)], [(-11.064126703880099,-6.2260195268943646), (-19.758820920288315,14.918813898717691), (2.484547501443704,-4.1730621607569827), (-2.3964329845298624,-11.866314511421749), (-9.1156214734948477,13.567490949817355), (-18.205864057503703,-3.1858427833865397)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

