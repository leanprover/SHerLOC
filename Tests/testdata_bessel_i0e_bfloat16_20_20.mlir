"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xbf16>
    %4 = "stablehlo.convert"(%2) : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %5 = "stablehlo.abs"(%4) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %6 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %8 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %10 = "stablehlo.constant"() <{value = dense<3.200000e+01> : tensor<f32>}> : () -> tensor<f32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %12 = "stablehlo.multiply"(%7, %5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %13 = "stablehlo.subtract"(%12, %9) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %15 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %18 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %19 = "stablehlo.broadcast_in_dim"(%18) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %20 = "stablehlo.multiply"(%13, %15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %21 = "stablehlo.subtract"(%20, %17) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "stablehlo.constant"() <{value = dense<-1.300025009986248E-8> : tensor<f64>}> : () -> tensor<f64>
    %23 = "stablehlo.convert"(%22) : (tensor<f64>) -> tensor<f32>
    %24 = "stablehlo.broadcast_in_dim"(%23) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %25 = "stablehlo.add"(%21, %24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %26 = "stablehlo.multiply"(%13, %25) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %27 = "stablehlo.subtract"(%26, %15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %28 = "stablehlo.constant"() <{value = dense<6.0469950225419186E-8> : tensor<f64>}> : () -> tensor<f64>
    %29 = "stablehlo.convert"(%28) : (tensor<f64>) -> tensor<f32>
    %30 = "stablehlo.broadcast_in_dim"(%29) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %31 = "stablehlo.add"(%27, %30) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.multiply"(%13, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "stablehlo.subtract"(%32, %25) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %34 = "stablehlo.constant"() <{value = dense<-2.6707938539406119E-7> : tensor<f64>}> : () -> tensor<f64>
    %35 = "stablehlo.convert"(%34) : (tensor<f64>) -> tensor<f32>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %37 = "stablehlo.add"(%33, %36) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %38 = "stablehlo.multiply"(%13, %37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %39 = "stablehlo.subtract"(%38, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %40 = "stablehlo.constant"() <{value = dense<1.1173875391201037E-6> : tensor<f64>}> : () -> tensor<f64>
    %41 = "stablehlo.convert"(%40) : (tensor<f64>) -> tensor<f32>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %43 = "stablehlo.add"(%39, %42) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %44 = "stablehlo.multiply"(%13, %43) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %45 = "stablehlo.subtract"(%44, %37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %46 = "stablehlo.constant"() <{value = dense<-4.4167383584587505E-6> : tensor<f64>}> : () -> tensor<f64>
    %47 = "stablehlo.convert"(%46) : (tensor<f64>) -> tensor<f32>
    %48 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %49 = "stablehlo.add"(%45, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.multiply"(%13, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.subtract"(%50, %43) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %52 = "stablehlo.constant"() <{value = dense<1.6448448070728896E-5> : tensor<f64>}> : () -> tensor<f64>
    %53 = "stablehlo.convert"(%52) : (tensor<f64>) -> tensor<f32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %55 = "stablehlo.add"(%51, %54) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %56 = "stablehlo.multiply"(%13, %55) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %57 = "stablehlo.subtract"(%56, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.constant"() <{value = dense<-5.754195010082104E-5> : tensor<f64>}> : () -> tensor<f64>
    %59 = "stablehlo.convert"(%58) : (tensor<f64>) -> tensor<f32>
    %60 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %61 = "stablehlo.add"(%57, %60) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %62 = "stablehlo.multiply"(%13, %61) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %63 = "stablehlo.subtract"(%62, %55) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<1.8850288509584165E-4> : tensor<f64>}> : () -> tensor<f64>
    %65 = "stablehlo.convert"(%64) : (tensor<f64>) -> tensor<f32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %67 = "stablehlo.add"(%63, %66) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %68 = "stablehlo.multiply"(%13, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.subtract"(%68, %61) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.constant"() <{value = dense<-5.7637557453858236E-4> : tensor<f64>}> : () -> tensor<f64>
    %71 = "stablehlo.convert"(%70) : (tensor<f64>) -> tensor<f32>
    %72 = "stablehlo.broadcast_in_dim"(%71) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %73 = "stablehlo.add"(%69, %72) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %74 = "stablehlo.multiply"(%13, %73) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %75 = "stablehlo.subtract"(%74, %67) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %76 = "stablehlo.constant"() <{value = dense<0.0016394756169413357> : tensor<f64>}> : () -> tensor<f64>
    %77 = "stablehlo.convert"(%76) : (tensor<f64>) -> tensor<f32>
    %78 = "stablehlo.broadcast_in_dim"(%77) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %79 = "stablehlo.add"(%75, %78) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %80 = "stablehlo.multiply"(%13, %79) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %81 = "stablehlo.subtract"(%80, %73) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %82 = "stablehlo.constant"() <{value = dense<-0.0043243099950505759> : tensor<f64>}> : () -> tensor<f64>
    %83 = "stablehlo.convert"(%82) : (tensor<f64>) -> tensor<f32>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %85 = "stablehlo.add"(%81, %84) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %86 = "stablehlo.multiply"(%13, %85) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %87 = "stablehlo.subtract"(%86, %79) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %88 = "stablehlo.constant"() <{value = dense<0.010546460394594998> : tensor<f64>}> : () -> tensor<f64>
    %89 = "stablehlo.convert"(%88) : (tensor<f64>) -> tensor<f32>
    %90 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %91 = "stablehlo.add"(%87, %90) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %92 = "stablehlo.multiply"(%13, %91) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %93 = "stablehlo.subtract"(%92, %85) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %94 = "stablehlo.constant"() <{value = dense<-0.023737414805899471> : tensor<f64>}> : () -> tensor<f64>
    %95 = "stablehlo.convert"(%94) : (tensor<f64>) -> tensor<f32>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %97 = "stablehlo.add"(%93, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %98 = "stablehlo.multiply"(%13, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %99 = "stablehlo.subtract"(%98, %91) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %100 = "stablehlo.constant"() <{value = dense<0.049305284239670712> : tensor<f64>}> : () -> tensor<f64>
    %101 = "stablehlo.convert"(%100) : (tensor<f64>) -> tensor<f32>
    %102 = "stablehlo.broadcast_in_dim"(%101) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %103 = "stablehlo.add"(%99, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %104 = "stablehlo.multiply"(%13, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %105 = "stablehlo.subtract"(%104, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %106 = "stablehlo.constant"() <{value = dense<-0.094901097048047639> : tensor<f64>}> : () -> tensor<f64>
    %107 = "stablehlo.convert"(%106) : (tensor<f64>) -> tensor<f32>
    %108 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %109 = "stablehlo.add"(%105, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %110 = "stablehlo.multiply"(%13, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %111 = "stablehlo.subtract"(%110, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %112 = "stablehlo.constant"() <{value = dense<0.17162090152220877> : tensor<f64>}> : () -> tensor<f64>
    %113 = "stablehlo.convert"(%112) : (tensor<f64>) -> tensor<f32>
    %114 = "stablehlo.broadcast_in_dim"(%113) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %115 = "stablehlo.add"(%111, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %116 = "stablehlo.multiply"(%13, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %117 = "stablehlo.subtract"(%116, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %118 = "stablehlo.constant"() <{value = dense<-0.3046826723431984> : tensor<f64>}> : () -> tensor<f64>
    %119 = "stablehlo.convert"(%118) : (tensor<f64>) -> tensor<f32>
    %120 = "stablehlo.broadcast_in_dim"(%119) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %121 = "stablehlo.add"(%117, %120) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %122 = "stablehlo.multiply"(%13, %121) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %123 = "stablehlo.subtract"(%122, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %124 = "stablehlo.constant"() <{value = dense<0.67679527440947607> : tensor<f64>}> : () -> tensor<f64>
    %125 = "stablehlo.convert"(%124) : (tensor<f64>) -> tensor<f32>
    %126 = "stablehlo.broadcast_in_dim"(%125) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %127 = "stablehlo.add"(%123, %126) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %128 = "stablehlo.subtract"(%127, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %129 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %130 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %131 = "stablehlo.multiply"(%130, %128) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %132 = "stablehlo.divide"(%11, %5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %133 = "stablehlo.subtract"(%132, %9) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %134 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %135 = "stablehlo.broadcast_in_dim"(%134) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %136 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %137 = "stablehlo.broadcast_in_dim"(%136) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %138 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %139 = "stablehlo.broadcast_in_dim"(%138) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %140 = "stablehlo.multiply"(%133, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %141 = "stablehlo.subtract"(%140, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %142 = "stablehlo.constant"() <{value = dense<3.3962320257083865E-9> : tensor<f64>}> : () -> tensor<f64>
    %143 = "stablehlo.convert"(%142) : (tensor<f64>) -> tensor<f32>
    %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %145 = "stablehlo.add"(%141, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %146 = "stablehlo.multiply"(%133, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %147 = "stablehlo.subtract"(%146, %135) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %148 = "stablehlo.constant"() <{value = dense<2.266668990498178E-8> : tensor<f64>}> : () -> tensor<f64>
    %149 = "stablehlo.convert"(%148) : (tensor<f64>) -> tensor<f32>
    %150 = "stablehlo.broadcast_in_dim"(%149) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %151 = "stablehlo.add"(%147, %150) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %152 = "stablehlo.multiply"(%133, %151) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %153 = "stablehlo.subtract"(%152, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %154 = "stablehlo.constant"() <{value = dense<2.0489185894690638E-7> : tensor<f64>}> : () -> tensor<f64>
    %155 = "stablehlo.convert"(%154) : (tensor<f64>) -> tensor<f32>
    %156 = "stablehlo.broadcast_in_dim"(%155) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %157 = "stablehlo.add"(%153, %156) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %158 = "stablehlo.multiply"(%133, %157) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %159 = "stablehlo.subtract"(%158, %151) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %160 = "stablehlo.constant"() <{value = dense<2.8913705208347567E-6> : tensor<f64>}> : () -> tensor<f64>
    %161 = "stablehlo.convert"(%160) : (tensor<f64>) -> tensor<f32>
    %162 = "stablehlo.broadcast_in_dim"(%161) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %163 = "stablehlo.add"(%159, %162) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %164 = "stablehlo.multiply"(%133, %163) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %165 = "stablehlo.subtract"(%164, %157) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %166 = "stablehlo.constant"() <{value = dense<6.8897583469168245E-5> : tensor<f64>}> : () -> tensor<f64>
    %167 = "stablehlo.convert"(%166) : (tensor<f64>) -> tensor<f32>
    %168 = "stablehlo.broadcast_in_dim"(%167) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %169 = "stablehlo.add"(%165, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %170 = "stablehlo.multiply"(%133, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %171 = "stablehlo.subtract"(%170, %163) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %172 = "stablehlo.constant"() <{value = dense<0.0033691164782556943> : tensor<f64>}> : () -> tensor<f64>
    %173 = "stablehlo.convert"(%172) : (tensor<f64>) -> tensor<f32>
    %174 = "stablehlo.broadcast_in_dim"(%173) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %175 = "stablehlo.add"(%171, %174) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %176 = "stablehlo.multiply"(%133, %175) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %177 = "stablehlo.subtract"(%176, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %178 = "stablehlo.constant"() <{value = dense<0.80449041101410879> : tensor<f64>}> : () -> tensor<f64>
    %179 = "stablehlo.convert"(%178) : (tensor<f64>) -> tensor<f32>
    %180 = "stablehlo.broadcast_in_dim"(%179) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %181 = "stablehlo.add"(%177, %180) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %182 = "stablehlo.subtract"(%181, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %183 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %184 = "stablehlo.broadcast_in_dim"(%183) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %185 = "stablehlo.multiply"(%184, %182) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %186 = "stablehlo.sqrt"(%5) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %187 = "stablehlo.divide"(%185, %186) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %188 = "stablehlo.constant"() <{value = dense<8.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %189 = "stablehlo.broadcast_in_dim"(%188) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %190 = "stablehlo.compare"(%5, %189) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %191 = "stablehlo.select"(%190, %131, %187) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %192 = "stablehlo.convert"(%191) : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    "stablehlo.custom_call"(%192, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    "func.return"(%192) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x7CC0DD40A8C0903C8ABF6D3F023F90C00DC0FF4021C01BC05740A83F5D3F92409FBD8B4035C065BFE33E87C06B4088C09A3F603E19C0BC4081BFC4BE4CBED6BEF03F85C009408FC02A4024C091BF6B402540094087BE84BF8440CCBF8BC0E5BF9CC0C4C0E0BF3EBF58BE203F9C3F0441B6C081BFD8C073402FC049BE94C00B40A1BFF83EB9401B3F863FC7BF1EC063BF51408DBF2DC05540B43FEBBE21C0A1BF57C094C095C0B1BFAF3F86C02EC00840AABE16C1CB407840E13E46C0243DCEBFF7C035C0A3BF10C08A40EA3E134014BF88BFAFC06A3F40C059C041C06E40E8C081C04540413F84C0CCC0EDBEA1403BC05140DABF954034BF68C056C07BC09ABF1C403A400E3F95C0484047BF2A409D3F6DC03EC095C020401A416ABFD13FD4C0463FFB3EA2C0854001C17040DC40DABFA1C0C83E8E40A13F9A3F12C019C02240E5BE61C032BE4DC09CBFD1407840D13F52BEACC06CC0DA3F4640523F1B40C4C0D6401740EB3FC9BE733FC83F573FA4BF31C022C082408D3FE6BE04BF8840D13F71C0EBBE00C03E3E0C401B40D33C8BBF71BE71BFA3BE9ABF40C0AB3F0ABF2E402E404A40014047401EC09040E2BFFFBFAEC0CD3E8340593F24BF823D5F400CC0D43F85BF6BC018C00FBF98BF02C015C0C040E8C04340C040B1C0913F754055C01C3F43C041C031C05DC04E4088BE4A401F3F2DC0873FA0BF2940814094BFD5BF82BF9DBFFABEA6BE5C3F39C0313F5BBFE740273F3A40FBBE31C03F4000C0363F273FC83FC1BE30401341DC401E40F93E4940D63EC73F81C004C05940EBC08F40C93E98BE86BF0F3F0B401FC01540A03FCBC0ABC00A408A3F1B4028C094C038C0B340AB3E1240DBBF083F13C05A3FCA3DCF3F863F84C09DC00DC11FC08040ACBFB4C00E4097BF01C0C8BF5DC0C1C0FA3F9FC0FDC0F93F93BFA94089404140254081C00340C83F553F633F8FC007BFCCBF843F10C027C0394087BF23403CBFAAC060BF103E8A3FA03E4A40A34022C08A40473E99C088BE3DBFA64015BDD23F76BFAE3AB9BDED3F12BD2CC0BDC07CC0B1BF6EC051BFE44098401F40D13EC4BE283F8C4048C015C01D4068C06A4068400A409840463FFABFABBF9C4033C0823F4840"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%1) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x563E1F3E373E7C3FE53EF93E243F473E953E133E8A3E8D3E693ECC3E013F453E6D3F4B3E813EFD3E2C3F4E3E5E3E4D3ED73E503F8E3E2D3EED3E353F543F303FA43E503E983E483E853E883EDE3E5E3E883E983E483FEA3E503EB53E4B3EA93E3F3E293EAB3E0B3F523F173FD53E113E303EED3E203E5A3E833E543F443E963ED13E273F2E3E193FE83EB83E8B3EFE3E6D3EE23E843E6B3EC43E2A3F8A3ED13E693E443E433EC63EC73E4F3E843E983E3D3F073E263E583E2D3F753E763FB43E163E813ED03E933E4B3E2B3F913E1C3FE73E333EFA3E793E683E783E5D3E1B3E533E753E0A3F503E253E2A3F3B3E7D3E6D3EAE3E433E0F3F603E6A3E563ED73E8C3E7D3E1F3F433E733E083F853ED43E5D3E7A3E433E8A3E053EFA3EB33E223E083F263F3B3E503E123E5C3E1F3EAE3E3B3E343F483ED13ED73E923E8E3E893E2C3F643E593F703ED53E233E583EB33E533F353E5E3EAE3E753E043F8D3E293E213E8F3EA63E343FF53EB83E033FCF3E823E893E523EE23E2C3F233F4D3EB33E5B3E2A3F9E3E563F963E8D3E7A3FE43E4D3FF63E3F3FD73E793ECA3E203F843E843E723E9D3E743E8B3E473EAA3E9E3E343E323F513E023F153F703F653E963EB13EE93E5E3E8F3E1E3FD83E9C3E903E2B3E1B3E773E2B3E323EDE3E593E6B3E183F773E783E823E663E6F3E483F723E173F843EE73ED23E863E533EDC3EB13EEC3ED43E273F3E3F013F7E3E103F023F1B3E143F7D3E263F823E7A3E9E3E0E3F143FB83E363F833E093E1F3E8B3E273F723E303FB83E533E9B3E683E1A3E483E343F423FE83E1E3F963E8B3E903ED23E263E353E973EE53E8D3E863E443E7F3E313E3C3F923EAE3E213F913E023F693FB43EE83E503E3E3E0C3E8B3E543EC93E313E943ED93E9D3EB83E663E2A3EA03E3D3E143EA13EDC3E373E4C3E783E883E533E9C3EB83E043FFE3E483E223FB53EEA3E933E873E7E3EE73E893E0C3F363E003F603FE53E403F723E3A3E893E4B3E553F403E483F0C3F383E773FB23EF43E803F6A3FA63E773F853E2C3E563EC63E5D3E053F1C3E413E8B3E313F353F133F4A3E733E903E8C3E603E5F3E603E973E413E083FA03ECA3E3F3E823EEC3E733E"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%0) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

