"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xcomplex<f32>>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xcomplex<f32>>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<8x9xcomplex<f32>>, tensor<8x9xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[(3.16425037,3.70685697), (0.586115181,-3.5972445), (-0.161650419,4.46904898), (1.02856326,-4.32734156), (0.0823069662,-0.192012206), (0.28745833,4.3364563), (-0.0161861554,4.79746246), (-3.37325907,3.15831614), (-0.0909689068,3.013120e+00)], [(3.62248206,9.426520e+00), (2.72655725,-1.8750205), (-2.35213137,1.84700239), (4.21684456,1.27245164), (1.5061332,-2.98743677), (-0.91299367,3.61682272), (2.60265017,0.0881813914), (-1.72490823,1.3182894), (-0.596892238,0.311736584)], [(-4.92009115,3.23211336), (-1.45702422,-7.88200426), (2.6062603,5.56625652), (1.26051974,7.493090e+00), (4.33715487,-3.35132599), (-3.25429296,-2.06495023), (-3.70643258,5.26480913), (-1.01719129,3.37386751), (1.79399645,-3.71674562)], [(-0.163486496,0.217105076), (2.00699615,-2.62414265), (-1.68265224,-2.269120e+00), (1.88508379,-3.69767666), (-1.66375673,-0.168235928), (-2.07322264,2.93306231), (1.4745301,-5.272730e+00), (2.11997771,1.33284497), (2.47433591,0.98611027)], [(-0.763706088,3.49935985), (-2.16241837,4.208340e+00), (-1.36019194,-0.471332908), (-3.47272944,-0.779990494), (0.0556155294,-1.93585193), (1.50940537,-1.72040069), (2.5159018,3.90656185), (-0.432686597,1.07842362), (1.10057831,-2.5701561)], [(1.36960042,-2.72286439), (-0.679398298,-0.906872212), (1.04589152,-1.90655661), (0.919022381,-2.6473527), (5.26230478,4.67007875), (-1.30459738,2.96472478), (1.6847614,-3.21171784), (2.4728384,-5.17069674), (0.984414875,0.00591267459)], [(-0.577216446,0.739891588), (-7.056260e-01,0.603553593), (2.39227033,0.361786693), (-2.81206608,-4.0171771), (-0.315536439,3.25484347), (-4.30015659,0.15144439), (0.329763949,-0.390107244), (-1.45133388,-4.72177362), (2.73322821,1.52241313)], [(-1.34346879,0.500030518), (4.38392258,-3.96865273), (5.66742611,8.370380e+00), (-3.8234036,-0.990289449), (-0.284813732,-2.60888147), (5.57516909,-2.39750242), (-3.34043622,-0.472224772), (1.10575771,-1.01911151), (3.75592351,-3.37539864)]]> : tensor<8x9xcomplex<f32>>}> : () -> tensor<8x9xcomplex<f32>>
    "func.return"(%4) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[(3.16425037,3.70685697), (0.586115181,-3.5972445), (-0.161650419,4.46904898), (1.02856326,-4.32734156), (0.0823069662,-0.192012206), (0.28745833,4.3364563), (-0.0161861554,4.79746246), (-3.37325907,3.15831614), (-0.0909689068,3.013120e+00)], [(-23.4803238,43.2558937), (-5.14683056,-10.9070711), (-7.87412119,-10.810359), (9.84362411,-16.9389305), (-0.44965905,-0.535082817), (-15.946641,-2.91947126), (-0.465173811,12.4846888), (1.65498769,-9.89473724), (-0.885001063,-1.82686627)], [(-24.2826195,-288.71402), (-78.4705276,56.4592094), (39.6512222,-72.0039902), (139.333023,52.4073067), (-3.74347782,-0.813782989), (45.866478,42.4298363), (-64.0053635,-48.7227097), (31.7000942,15.64855), (-8.37768555,0.0119321989)], [(66.6511612,41.9289627), (-9.33303451,319.231293), (-230.104919,31.1842937), (456.439697,-416.416321), (6.0913291,1.98372436), (-219.540771,46.5627403), (-351.27951,265.639893), (4.634640e+01,75.425888), (-20.7409744,-8.23179817)], [(-197.626434,2.012150e+02), (-1323.25195,-729.588195), (327.685028,6.603940e+01), (-1909.89233,1090.08264), (4.17896891,-11.6815853), (-251.269455,447.980133), (-1921.52344,-703.971252), (-101.394524,17.3452816), (-43.9840736,44.2478027)], [(277.211914,813.694152), (237.371872,1695.70142), (468.630859,-5.556800e+02), (1130.59949,6057.96923), (76.5449295,-41.9559479), (-1000.33234,-1329.37842), (-5498.26563,4985.36768), (-161.04509,567.172424), (-4.356020e+01,43.29813)], [(-762.056763,-264.570892), (-1190.94238,-1053.2644), (1322.12927,-1159.79236), (21156.6152,-21577.2285), (112.407326,262.380402), (4502.9126,5565.04053), (131.698212,3788.90796), (2.911790e+03,-6.273810e+01), (-184.9776,52.02705)], [(1156.09302,-2.560890e+01), (-9.401040e+03,109.00692), (17200.9727,4493.68701), (-102257.977,61547.2773), (652.504211,-367.986938), (3.844670e+04,20230.2969), (1349.28674,-12718.7969), (3155.79712,-3036.81177), (-519.149658,819.782775)]]> : tensor<8x9xcomplex<f32>>}> : () -> tensor<8x9xcomplex<f32>>
    "func.return"(%3) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xcomplex<f32>>) -> tensor<8x9xcomplex<f32>>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xcomplex<f32>>):
    %0 = "stablehlo.constant"() <{value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
      "stablehlo.return"(%2) : (tensor<complex<f32>>) -> ()
    }) : (tensor<8x9xcomplex<f32>>, tensor<complex<f32>>) -> tensor<8x9xcomplex<f32>>
    "func.return"(%1) : (tensor<8x9xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

